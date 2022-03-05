import json
import torch
import genova
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

with open('genova/utils/dictionary') as f:
    dictionary = json.load(f)

cfg = OmegaConf.load('configs/genova_dda_light.yaml')
spec_header = pd.read_csv('/data/z37mao/genova/pretrain_data_sparse/genova_psm.csv',index_col='index')
spec_header = spec_header[np.logical_or(spec_header['Experiment Name']=='Cerebellum',spec_header['Experiment Name']=='HeLa')]
small_spec = spec_header[spec_header['Node Number']<=256]
dataset = genova.data.GenovaDataset(cfg,dictionary=dictionary,spec_header=small_spec,dataset_dir_path='/data/z37mao/genova/pretrain_data_sparse/')
collate_fn  = genova.data.GenovaCollator(cfg,mode='train')
dl = DataLoader(dataset,batch_size=4,collate_fn=collate_fn,num_workers=8,prefetch_factor=4,shuffle=True)

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
local_rank = int(os.environ['LOCAL_RANK'])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 构造模型
device = torch.device("cuda", local_rank)
model = genova.models.Genova(cfg).to(local_rank)
loss_fn = nn.CrossEntropyLoss()

checkpoint = torch.load('/data/z37mao/save/GenovaPrototype.pt',map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank})
model.load_state_dict(checkpoint)
#for param in model.encoder.parameters(): param.requires_grad = False
model = DDP(model, device_ids=[local_rank])

optimizer = optim.AdamW(model.parameters(),lr=3e-5)
scaler = GradScaler()

def encoder_input_cuda(encoder_input,device):
    for section_key in encoder_input:
        for key in encoder_input[section_key]:
            if isinstance(encoder_input[section_key][key],torch.Tensor):
                encoder_input[section_key][key] = encoder_input[section_key][key].to(device)
    return encoder_input

def decoder_input_cuda(decoder_input,device):
    for key in decoder_input:
        if isinstance(decoder_input[key],torch.Tensor):
            decoder_input[key] = decoder_input[key].to(device)
    return decoder_input

torch.cuda.empty_cache()


loss_detect = 0
min_loss = 10000
detect_period = 200
for epoch in range(5):
    for i, (encoder_input, decoder_input, labels) in enumerate(dl,start=1):
        encoder_input = encoder_input_cuda(encoder_input,device)
        decoder_input = decoder_input_cuda(decoder_input,device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(encoder_input=encoder_input,decoder_input=decoder_input)
            loss = loss_fn(output[labels!=0],labels[labels!=0])
        if local_rank==0:
            loss_detect+=loss.item()
            if i%detect_period==0:
                loss_ave = loss_detect/detect_period
                print(loss_ave)
                if min_loss>loss_ave:
                    min_loss = loss_ave
                    torch.save({'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},'/data/z37mao/save/Genova_model.pt')
                loss_detect = 0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()