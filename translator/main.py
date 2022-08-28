import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
from BasicClass import Residual_seq, Ion
from torch.nn.functional import one_hot

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from Loss import FocalLoss
from Sampler import PERSampler
from ReplayMemory import ReplayMemoryDataset
from prefetcher import DataPrefetcher, DataTransfer
from Environment import EnvironmentDataset, Env_Interactor
from GenovaTranslator import GenovaTranslator
from data import ValidDataset

data_df_train = pickle.load(open('/home/z37mao/translator_data/translator_dataset_selected.mgfs.pt','rb'))
psm_head_train = pd.read_pickle('/home/z37mao/translator_data/train_ds.mgfsheader')

data_df_val = pickle.load(open('/home/z37mao/translator_data/translator_valid.mgfs.pt','rb'))
psm_head_val = pd.read_pickle('/home/z37mao/translator_data/translate_valid_PSMs.mgfsheader')
psm_head_val = psm_head_val.loc[psm_head_val['Peaks Number']<256]

aa_dict = {aa:i for i, aa in enumerate(Residual_seq.output_aalist(),start=4)}
aa_dict['<pad>'] = 0
aa_dict['<bos>'] = 1
aa_dict['<x>'] = 2
aa_dict['<answer>'] = 3

tokenize_aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aalist())}
detokenize_aa_dict = {i: aa for i, aa in enumerate(Residual_seq.output_aalist())}

knapsack = pickle.load(open('knapsack.pt','rb'))
knapsack_edge = pickle.load(open('knapsack_3000.pt','rb'))

def train(model, train_loader, val_loader, optimizer, scaler, training_loss_fn, inference_loss_fn, rank, local_rank, scheduler=None):
    total_step = 0
    train_loss = 0.0
    detect_period = 100
    reward = 0
    candidate_token_number = 0
    min_loss = float('inf')

    for idx, model_input, label in train_loader:
        model.train()
        label, label_mask = label['label'], label['label_mask']
        optimizer.zero_grad()
        with autocast():
            outputs = model(**model_input)
            loss = training_loss_fn(outputs, label, idx=idx, local_rank=local_rank, label_mask=label_mask)
            max_loss = inference_loss_fn(outputs.detach(), label, label_mask=label_mask, reduction='max')
        reward += label[:16][label_mask[:16]][:,:20].sum().item()
        candidate_token_number += label_mask[:16].sum().item()
        train_loader.loader_ori.dataset.memory_loss_update(idx,max_loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if rank == 0:
            train_loss += loss.item()
            if total_step % detect_period == 0 and total_step > 0:
                print(f"Step {total_step}: loss {train_loss/detect_period} reward {reward/candidate_token_number} len {candidate_token_number/(detect_period*16)}")
                train_loss = 0
                reward = 0
                candidate_token_number = 0
            
        if total_step % (20*detect_period) == 0 and total_step > 0:
            eval_loss, total_token_num, accuracy = eval(model, val_loader, inference_loss_fn, local_rank)
            dist.barrier()
            dist.all_reduce(eval_loss)
            dist.all_reduce(total_token_num)
            eval_loss = eval_loss.item() / total_token_num.item()
            accuracy = accuracy.item() / total_token_num.item()
            if scheduler!=None: scheduler.step(eval_loss)

            if rank == 0:
                print(f"Eval loss: {eval_loss}")
                torch.save({'model_state_dict':model.module.state_dict(),
                            'optimizer_state_dict':optimizer.state_dict()},
                            './save/rl_model.pt')
                if eval_loss < min_loss:
                    torch.save({'model_state_dict':model.module.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict()},
                                './save/rl_model_best.pt')
                    min_loss = eval_loss
                    print(f"New Minimum Eval loss: {min_loss} Accuracy: {accuracy}")

        total_step += 1


def eval(model, val_loader, loss_fn, local_rank):
    model.eval()
    with torch.no_grad():
        accuracy = torch.tensor(0.0).to(local_rank)
        eval_loss = torch.tensor(0.0).to(local_rank)
        total_token_num = torch.tensor(0).to(local_rank)
        for model_input, label in val_loader:
            model.train()
            label, label_mask = label['label'], label['label_mask']
            with autocast(): 
                outputs = model(**model_input)
                accuracy += (outputs[...,:20].argmax(-1)==label[...,:20].argmax(-1))[label_mask].sum()
                loss = loss_fn(outputs, label, label_mask, reduction='sum')
            token_num = label_mask.sum()
            total_token_num += token_num
            eval_loss += loss.sum()
    return eval_loss, total_token_num, accuracy

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

inference_batch_size = 16
train_batch_size = 32
val_batch_size = 128
memory_pool_capacity = 2**8
#memory_pool_capacity = 16

inference_loss_fn = FocalLoss()
model = GenovaTranslator(768,12).to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

#map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
#checkpoint = torch.load('./save/rl_model.pt', map_location=map_location)
#model.module.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.2)
scaler = GradScaler()

env_ds = EnvironmentDataset(aa_dict, data_df_train, psm_head_train)
env_dds = DistributedSampler(env_ds, num_replicas=world_size, rank=rank)
env_dl = DataLoader(env_ds, batch_size=inference_batch_size, num_workers=8, sampler=env_dds, pin_memory=True)
inference_dl = DataPrefetcher(env_dl,local_rank,valid=False)
env_interactor = Env_Interactor(knapsack, knapsack_edge, aa_dict, tokenize_aa_dict, detokenize_aa_dict, psm_head_train, local_rank, model.module, inference_dl, inference_loss_fn, ms1_threshold_ppm=40, ms2_threshold_da=0.04)
replaymemory = ReplayMemoryDataset(memory_pool_capacity, env_interactor, aa_dict, data_df_train, psm_head_train, tokenize_aa_dict)
sampler = PERSampler(replaymemory, train_batch_size, inference_batch_size)
train_dl = DataLoader(dataset=replaymemory, batch_sampler=sampler)
train_dl = DataTransfer(train_dl, local_rank)
train_loss_fn = FocalLoss(training=True, sampler=sampler, batch_size=inference_batch_size)

val_ds = ValidDataset(aa_dict, data_df_val, psm_head_val, tokenize_aa_dict)
val_dds = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=8, sampler=val_dds, pin_memory=True)
val_dl = DataPrefetcher(val_dl,local_rank,valid=True)

train(model, train_dl, val_dl, optimizer, scaler, train_loss_fn, inference_loss_fn, rank, local_rank)
