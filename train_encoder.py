import json
from threading import local
from wsgiref.simple_server import WSGIRequestHandler
import torch
import genova
from datetime import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from genova.utils.BasicClass import Residual_seq
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.nn as nn
import torch.optim as optim

import wandb

cfg = OmegaConf.load('configs/genova_dda_light.yaml')
spec_header = pd.read_csv('/data/z37mao/genova/pretrain_data_sparse/genova_psm.csv',index_col='index')
spec_header = spec_header[np.logical_or(spec_header['Experiment Name']=='Cerebellum',spec_header['Experiment Name']=='HeLa')]
small_spec = spec_header[spec_header['Node Number']<=256]

import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset
import numpy as np
from torch.nn.functional import pad

class GenovaCollator(object):
    def __init__(self,cfg):
        self.cfg = cfg
        
    def __call__(self,batch):
        encoder_records = [record[0] for record in batch]
        labels_ori = [record[1] for record in batch]
        encoder_input, node_mask = self.encoder_collate(encoder_records)
        max_node = max([label.shape[0] for label in labels_ori])
        labels = []
        for label_ori in labels_ori:
            labels.append(pad(label_ori,[0,max_node-label_ori.shape[0]]))
        labels = torch.stack(labels)
        
        return encoder_input, labels, node_mask
        
    def encoder_collate(self, encoder_records):
        node_shape = []
        for record in encoder_records: node_shape.append(np.array(record['node_sourceion'].shape))
        node_shape = np.array(node_shape).T
        max_node = node_shape[0].max()
        max_subgraph_node = node_shape[1].max()

        node_input = {}
        edge_input = {}
        rel_input = {}

        edge_input['rel_type'] = torch.concat([record['rel_type'] for record in encoder_records])
        edge_input['edge_pos'] = torch.concat([record['edge_pos'] for record in encoder_records])
        edge_input['rel_error'] = torch.concat([record['rel_error'] for record in encoder_records]).unsqueeze(-1)


        node_feat = []
        node_sourceion = []
        rel_mask = []
        dist = []
        charge = []
        rel_coor_cated = []
        node_mask = torch.zeros(len(encoder_records),max_node,dtype=bool)
        for i, record in enumerate(encoder_records):
            node_num, node_subgraph_node = record['node_sourceion'].shape
            node_feat.append(pad(record['node_feat'],[0,0,0,max_subgraph_node-node_subgraph_node,0,max_node-node_num]))
            node_sourceion.append(pad(record['node_sourceion'],[0,max_subgraph_node-node_subgraph_node,0,max_node-node_num]))
            rel_mask.append(pad(pad(record['rel_mask'],[0,max_node-node_num],value=-float('inf')),[0,0,0,max_node-node_num]))
            dist.append(pad(record['dist'],[0,max_node-node_num,0,max_node-node_num]))
            charge.append(record['charge'])
            rel_coor_cated.append(torch.stack([i*max_node**2+record['rel_coor'][0]*max_node+record['rel_coor'][1],
                                               record['rel_coor'][-2]*100+record['rel_coor'][-1]]))
            node_mask[i,node_num:] = True

        drctn = torch.zeros(max_node,max_node)+torch.tril(2*torch.ones(max_node,max_node),-1)+torch.triu(torch.ones(max_node,max_node),1)
        rel_input['drctn'] = drctn.int().unsqueeze(0)
        node_input['node_feat'] = torch.stack(node_feat)
        node_input['node_sourceion'] = torch.stack(node_sourceion)
        rel_input['rel_mask'] = torch.stack(rel_mask).unsqueeze(-1)
        edge_input['dist'] = torch.stack(dist)
        node_input['charge'] = torch.IntTensor(charge)
        edge_input['rel_coor_cated'] = torch.concat(rel_coor_cated,dim=1)
        edge_input['batch_num'] = len(encoder_records)
        edge_input['max_node'] = max_node
        
        encoder_input = {'node_input':node_input,'edge_input':edge_input,'rel_input':rel_input}

        return encoder_input, node_mask

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        #self.dictionary = dictionary
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.iloc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['Serialized File Name']),'rb') as f:
            f.seek(spec_head['Serialized File Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['Serialized Data Length'])))
        
        spec['charge'] = spec_head['Charge']
        label = spec.pop('path_label')
        label = torch.any(label,-1).long()
        edge_type = spec.pop('edge_type')
        edge_error = spec.pop('edge_error')
        #edge_coor = torch.stack(torch.where(edge_type>0))
        #edge_error = edge_error[edge_type>0]
        #edge_type = edge_type[edge_type>0]
        return spec, label
        
    def __len__(self):
        return len(self.spec_header)

def encoder_input_cuda(encoder_input, device):
    for section_key in encoder_input:
        for key in encoder_input[section_key]:
            if isinstance(encoder_input[section_key][key],torch.Tensor):
                encoder_input[section_key][key] = encoder_input[section_key][key].to(device)
    return encoder_input


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
local_rank = int(os.environ['LOCAL_RANK'])

if local_rank==0:
    wandb.init(project="Genova", entity="amadeusandiris")

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)

ds = GenovaDataset(cfg, spec_header=small_spec, dataset_dir_path='/data/z37mao/genova/pretrain_data_sparse/')
collate_fn = GenovaCollator(cfg)
dl = DataLoader(ds,batch_size=4,collate_fn=collate_fn,num_workers=4,shuffle=True)
model = genova.GenovaEncoder(cfg,bin_classification=True).to(local_rank)
model = DDP(model, device_ids=[local_rank])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=2e-4)
scaler = GradScaler()

CHECKPOINT_PATH = '/data/z37mao/genova/save/model_max.pt'
#checkpoint = torch.load(CHECKPOINT_PATH,map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank})['model_state_dict']
checkpoint = torch.load(CHECKPOINT_PATH,map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank})
if list(model.state_dict().keys())[0].startswith('module'):
    #model.load_state_dict(checkpoint)
    model.load_state_dict(OrderedDict([('module.'+key, v) for key, v in checkpoint.items()]))
else:
    #model.load_state_dict(OrderedDict([(key[7:], v) for key, v in checkpoint.items()]))
    model.load_state_dict(checkpoint)

loss_detect = 0
min_loss = 10000
detect_period = 200
accuracy = 0
recall = 0
precision = 0
for epoch in range(5):
    for i, (encoder_input, labels, node_mask) in enumerate(dl,start=1):
        encoder_input = encoder_input_cuda(encoder_input,device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(**encoder_input)
            loss = loss_fn(output[~node_mask],labels[~node_mask])
        if local_rank==0:
            output = torch.argmax(output[~node_mask],-1)
            labels = labels[~node_mask]
            accuracy += (output==labels).sum()/labels.shape[0]
            recall += ((output==labels)[labels==1]).sum()/(labels==1).sum()
            precision += ((output==labels)[labels==1]).sum()/(output==1).sum()
            loss_detect += loss.item()
            if i%detect_period==0:
                wandb.log({"loss": loss_detect/detect_period, 
                "accuracy": accuracy/detect_period, 
                "recall": recall/detect_period, 
                "precision": precision/detect_period}
                )
                torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()},'/data/z37mao/genova/save/model_max.pt')
                loss_detect, accuracy, recall, precision = 0, 0, 0, 0
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
