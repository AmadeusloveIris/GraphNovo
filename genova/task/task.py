import os
import torch
import genova
from torch import nn, optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .optimal_path_inference import optimal_path_infer
from .seq_generation_inference import seq_generation_infer

class Task:
    def __init__(self, cfg, serialized_model_path, distributed=True):
        self.cfg = cfg
        self.distributed = distributed
        self.serialized_model_path = serialized_model_path
        if cfg.mode == 'train':
            if self.distributed:
                dist.init_process_group(backend='nccl')
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.device = torch.device("cuda", self.local_rank)
                torch.cuda.set_device(self.local_rank)
            else: self.device = torch.device('cuda')
        else:
            if isinstance(cfg.infer.device, int):
                torch.cuda.set_device(cfg.infer.device)
                self.device = torch.device('cuda:'+str(cfg.infer.device))
            else:
                self.device = torch.device('cpu')

    def initialize(self, *, train_spec_header,train_dataset_dir,val_spec_header,val_dataset_dir):
        self.model = genova.models.Genova(self.cfg).to(self.device)
        
        if self.cfg.task == 'optimal_path':
            self.train_loss_fn = nn.KLDivLoss(reduction='batchmean')
            self.eval_loss_fn = nn.KLDivLoss(reduction='sum')
        elif self.cfg.task == 'node_classification':
            self.train_loss_fn = nn.BCEWithLogitsLoss()
            self.eval_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
        elif self.cfg.task == 'sequence_generation':
            self.train_loss_fn = nn.CrossEntropyLoss()
            self.eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        else:
            raise NotImplementedError
        
        assert self.distributed==dist.is_initialized()
        if self.distributed: self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.train.lr)
        self.scaler = GradScaler()
        self.persistent_file_name = os.path.join(self.serialized_model_path,self.cfg.wandb.project+'_'+self.cfg.wandb.name+'.pt')
        if os.path.exists(self.persistent_file_name):
            checkpoint = torch.load(self.persistent_file_name)
            if self.distributed: self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else: self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.train_dl = self.train_loader(train_spec_header,train_dataset_dir)
        self.eval_dl = self.eval_loader(val_spec_header,val_dataset_dir)

    def test_initialize(self, *, test_spec_header=None, test_dataset_dir=None):
        assert not self.distributed
        self.model = genova.models.Genova(self.cfg).to(self.device)
        self.persistent_file_name = os.path.join(self.serialized_model_path,self.cfg.wandb.project+'_'+self.cfg.wandb.name+'.pt')
        print('checkpoint: ', self.persistent_file_name)
        assert os.path.exists(self.persistent_file_name)
        if isinstance(self.cfg.infer.device, int):
            checkpoint = torch.load(self.persistent_file_name)
        else:
            checkpoint = torch.load(self.persistent_file_name,map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.test_dl = self.test_loader(test_spec_header,test_dataset_dir)
        self.test_spec_header = test_spec_header

    def train_loader(self,train_spec_header,train_dataset_dir):
        ds = genova.data.GenovaDataset(self.cfg,spec_header=train_spec_header,dataset_dir_path=train_dataset_dir)
        sampler = genova.data.GenovaBatchSampler(self.cfg,self.device,0.95,train_spec_header,[0,128,256,512],self.model)
        collate_fn = genova.data.GenovaCollator(self.cfg)
        if self.distributed:
            train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=10)
        else:
            train_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        train_dl = genova.data.DataPrefetcher(train_dl,self.device)
        return train_dl

    def eval_loader(self,val_spec_header,val_dataset_dir):
        ds = genova.data.GenovaDataset(self.cfg,spec_header=val_spec_header,dataset_dir_path=val_dataset_dir)
        sampler = genova.data.GenovaBatchSampler(self.cfg,self.device,2,val_spec_header,[0,128,256,512],self.model)
        collate_fn = genova.data.GenovaCollator(self.cfg)
        if self.distributed:
            eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True,num_workers=5)
        else:
            eval_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        eval_dl = genova.data.DataPrefetcher(eval_dl,self.device)
        return eval_dl
    
    def test_loader(self,test_spec_header,test_dataset_dir):
        ds = genova.data.GenovaDataset(self.cfg,spec_header=test_spec_header,dataset_dir_path=test_dataset_dir)
        sampler = genova.data.GenovaSequentialSampler(test_spec_header)
        collate_fn = genova.data.GenovaCollator(self.cfg)
        test_dl = DataLoader(ds,batch_sampler=sampler,collate_fn=collate_fn,pin_memory=True)
        if isinstance(self.cfg.infer.device, int):
            test_dl = genova.data.DataPrefetcher(test_dl,self.device)
        return test_dl
    
    def model_save(self):
        if self.distributed:
            torch.save({'model_state_dict':self.model.module.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()},self.persistent_file_name)
        else:
            torch.save({'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict()},self.persistent_file_name)

    def train(self):
        total_step = 0
        loss_cum = 0
        if self.cfg.task =='node_classification':
            for epoch in range(0, self.cfg.train.total_epoch):
                for encoder_input, label, label_mask in self.train_dl:
                    total_step += 1
                    if total_step%self.cfg.train.detect_period == 1: loss_cum = 0
                    self.optimizer.zero_grad()
                    with autocast():
                        output = self.model(encoder_input=encoder_input).squeeze(-1)
                        loss = self.train_loss_fn(output[label_mask],label[label_mask])
                    loss_cum += loss.item()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if total_step%self.cfg.train.detect_period == 0: yield loss_cum/self.cfg.train.detect_period, total_step, epoch
        else:
            for epoch in range(0, self.cfg.train.total_epoch):
                for encoder_input, decoder_input, tgt, label, label_mask, _ in self.train_dl:
                    total_step += 1
                    if total_step%self.cfg.train.detect_period == 1: loss_cum = 0
                    self.optimizer.zero_grad()
                    with autocast():
                        output = self.model(encoder_input=encoder_input, decoder_input=decoder_input, tgt=tgt)
                        if self.cfg.task == 'optimal_path': output = output.log_softmax(-1)
                        loss = self.train_loss_fn(output[label_mask],label[label_mask])
                    loss_cum += loss.item()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if total_step%self.cfg.train.detect_period == 0: yield loss_cum/self.cfg.train.detect_period, total_step, epoch

    def eval(self) -> float:
        loss_cum = torch.Tensor([0]).to(self.device)
        total_seq_len = torch.Tensor([0]).to(self.device)
        if self.cfg.task =='node_classification':
            total_match = torch.Tensor([0]).to(self.device)
            true_positive = torch.Tensor([0]).to(self.device)
            total_positive = torch.Tensor([0]).to(self.device)
            total_true = torch.Tensor([0]).to(self.device)
            for encoder_input, label, label_mask in self.eval_dl:
                with torch.no_grad():
                    with autocast():
                        output = self.model(encoder_input=encoder_input)
                        output = output[label_mask].squeeze(-1)
                        label = label[label_mask]
                        loss = self.eval_loss_fn(output,label)
                    output = (output>0.5).float()
                    loss_cum += loss
                    total_seq_len += label_mask.sum()
                    total_match += (output == label).sum()
                    true_positive += ((output == label)[label == 1]).sum()
                    total_positive += (label == 1).sum()
                    total_true += (output == 1).sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
                dist.all_reduce(total_match)
                dist.all_reduce(true_positive)
                dist.all_reduce(total_positive)
                dist.all_reduce(total_true)
            return (loss_cum/total_seq_len).item(), \
                   (total_match/total_seq_len).item(), \
                   (true_positive/total_positive).item(), \
                   (true_positive/total_true).item()
        else:
            for encoder_input, decoder_input, tgt, label, label_mask, _ in self.eval_dl:
                with torch.no_grad():
                    with autocast():
                        output = self.model(encoder_input=encoder_input, decoder_input=decoder_input, tgt=tgt)
                        if self.cfg.task == 'optimal_path': output = output.log_softmax(-1)
                        loss = self.eval_loss_fn(output[label_mask],label[label_mask])
                    loss_cum += loss
                    total_seq_len += label_mask.sum()
            if self.distributed:
                dist.barrier()
                dist.all_reduce(loss_cum)
                dist.all_reduce(total_seq_len)
            return (loss_cum/total_seq_len).item()
        
    def inference(self) -> float:
        if self.cfg.task == 'optimal_path':
            optimal_path_infer(self.cfg, self.test_spec_header, self.test_dl, self.model, self.device)
        elif self.cfg.task == 'sequence_generation':
            seq_generation_infer(self.cfg, self.test_spec_header, self.test_dl, self.model, self.device)
