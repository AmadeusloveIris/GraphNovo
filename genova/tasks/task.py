import os
import wandb
import torch
import genova
from omegaconf import OmegaConf
from torch import nn, optim
import torch.distributed as dist
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

class Task:
    def __init__(self, cfg, model_save_dir, distributed=True):
        self.cfg = cfg
        self.distributed = distributed
        self.model_save_dir = model_save_dir
        if self.distributed:
            dist.init_process_group(backend='nccl')
            self.device = torch.cuda.device(int(os.environ["LOCAL_RANK"]))
        else: self.device = torch.cuda.device('cuda') 

    def initialize(self,train_spec_header,train_dataset_dir,val_spec_header,val_dataset_dir):
        self.model = genova.models.Genova(self.cfg)
        self.train_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.eval_loss_fn = nn.KLDivLoss(reduction='sum')
        self.optimizer = optim.AdamW(self.model_ori.parameters(), lr=self.cfg.train.lr)
        self.scaler = GradScaler()
        persistent_file_name = os.path.join(self.model_save_dir,self.cfg.wandb.project+'.pt')
        if os.path.exists(persistent_file_name):
            checkpoint = torch.load(persistent_file_name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_dl = self.train_loader(train_spec_header,train_dataset_dir)
        self.eval_dl = self.eval_loader(val_spec_header,val_dataset_dir)
        
        assert self.distributed==dist.is_initialized()
        if self.distributed: self.model = DDP(self.model, device_ids=[self.device])

        wandb.init(entity=self.cfg.wandb.entity, project=self.cfg.wandb.project)
        wandb.config = OmegaConf.to_container(self.cfg)
        wandb.watch(self.model,log='all')

    def train_loader(self,train_spec_header,train_dataset_dir):
        ds = genova.data.GenovaDataset(self.cfg,spec_header=train_spec_header,dataset_dir_path=train_dataset_dir)
        sampler = genova.data.GenovaBatchSampler(self.cfg,self.device,0.95,train_spec_header,[0,128,256,512],self.model_ori)
        collate_fn = genova.data.GenovaCollator(self.cfg)
        train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=(cpu_count()-1)//4,prefetch_factor=4)
        train_dl = genova.data.DataPrefetcher(train_dl,self.device)
        return train_dl

    def eval_loader(self,val_spec_header,val_dataset_dir):
        ds = genova.data.GenovaDataset(self.cfg,spec_header=val_spec_header,dataset_dir_path=val_dataset_dir)
        sampler = genova.data.GenovaBatchSampler(self.cfg,self.device,2,val_spec_header,[0,128,256,512,768],self.model_ori)
        collate_fn = genova.data.GenovaCollator(self.cfg)
        eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=(cpu_count()-1)//4)
        eval_dl = genova.data.DataPrefetcher(eval_dl,self.device)
        return eval_dl

    def train(self):
        total_step = 0
        for epoch in range(0, self.cfg.train.total_epoch):
            for encoder_input, decoder_input, graph_probability, label, label_mask in self.train_dl:
                total_step += 1
                if total_step%self.cfg.train.detect_period == 1: loss_cum = 0
                self.optimizer.zero_grad()
                with autocast():
                    output = self.model(encoder_input=encoder_input, decoder_input=decoder_input, graph_probability=graph_probability)
                    output = output.log_softmax(-1)
                    loss = self.train_loss_fn(output[label_mask],label[label_mask])
                assert loss.item()!=float('nan')
                loss_cum += loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if total_step%self.cfg.train.detect_period == 0: yield loss_cum/self.cfg.train.detect_period, total_step

    def eval(self) -> float:
        loss_cum = 0
        total_seq_len = 0
        for encoder_input, decoder_input, graph_probability, label, label_mask in self.eval_dl:
            with torch.no_grad():
                output = self.model(encoder_input=encoder_input, decoder_input=decoder_input, graph_probability=graph_probability)
                output = output.log_softmax(-1)
                loss = self.eval_loss_fn(output[label_mask],label[label_mask])
            assert loss.item()!=float('nan')
            loss_cum += loss
            total_seq_len += label_mask.sum()
        return loss_cum, total_seq_len