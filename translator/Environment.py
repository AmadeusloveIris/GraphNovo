import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.functional import pad
from BasicClass import Residual_seq, Ion
from random import random

import collections
from torch._six import string_classes
from dataclasses import dataclass

@dataclass
class Pep_Inference_Status:
    idx: int
    inference_seq: list[str]
    current_mass: float
    current_end_mass: float
    current_pepblock_index: int
    current_aaindex: int = 0
    current_pepblock_count: int = 0
    edge_true: bool = True
    pep_true: bool = True
    max_loss: float = 0

@dataclass
class Pep_Finish_Status:
    idx: int
    train_seq: str
    max_loss: float

@dataclass
class Edge_Info:
    edge_seq: list[str]
    edge_index: int
    edge_start_mass: float
    edge_end_mass: float

class Env_Interactor(object):
    def __init__(self, knapsack, knapsack_edge, aa_dict, tokenize_aa_dict, detokenize_aa_dict, psm_head, local_rank, model, inference_dl, loss_fn, hidden_size=768, nhead=12, ms1_threshold_ppm=5, ms2_threshold_da=0.02):
        self.knapsack_mass = knapsack['mass']
        self.knapsack_mass_len = knapsack['mass'].size
        self.knapsack_index = knapsack['indexes']
        self.knapsack_aa_composition = knapsack['aa_composition']

        self.knapsack_edge_mass = knapsack_edge['mass']
        self.knapsack_edge_mass_len = knapsack_edge['mass'].size
        self.knapsack_edge_aa_composition = knapsack_edge['aa_composition']

        self.single_aa_mask_list = ['N','Q','K','m','F','R','W']
        
        self.aa_dict = aa_dict
        self.tokenize_aa_dict = tokenize_aa_dict
        self.detokenize_aa_dict = detokenize_aa_dict

        self.psm_head = psm_head
        self.local_rank = local_rank

        self.ms1_threshold_ppm = ms1_threshold_ppm
        self.ms2_threshold_da = ms2_threshold_da
        self.inference_dl_ori = inference_dl

        self.hidden_size = hidden_size
        self.nhead = nhead
        self.model = model
        self.loss = loss_fn
    
    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        pep_finish_pool = self.update_replay_buffer()
        return pep_finish_pool

    def update_replay_buffer(self):
        try:
            src, encoder_input, tgt, tgt_mask, pep_mass, mem_key_padding_mask, peaks_moverz, idx  = next(self.inference_dl)
        except StopIteration:
            self.inference_dl = iter(self.inference_dl_ori)
            src, encoder_input, tgt, tgt_mask, pep_mass, mem_key_padding_mask, peaks_moverz, idx  = next(self.inference_dl)
        self.model.eval()
        mem = self.model.mem_get(src, encoder_input)
        self.model.inference_initial()
        past_tgts, past_mems = self.model.inference_step(tgt=tgt, tgt_mask=tgt_mask, mem=mem, pep_mass=pep_mass, mem_key_padding_mask=mem_key_padding_mask, peaks_moverz=peaks_moverz)

        idx = idx.tolist()
        seqs = self.psm_head.iloc[idx]['Annotated Sequence']
        edges_info = [self.edge_info(seq) for seq in seqs]
        precursor_mass = np.array(Ion.precursorion2mass(self.psm_head.iloc[idx]['m/z [Da]'],self.psm_head.iloc[idx]['Charge']))
        edge_mass_threshold = precursor_mass*self.ms1_threshold_ppm*1e-6+self.ms2_threshold_da*2
        pep_mass_threshold = precursor_mass*self.ms1_threshold_ppm*1e-6
        pep_status_list = self.pep_status_initializer(edges_info, idx)
        pep_finish_pool = []

        while len(pep_status_list)>0:
            tgt, pep_mass, knapsack_mask = self.model_input_generator(pep_status_list, precursor_mass, pep_mass_threshold, edge_mass_threshold)
            tgt_mask = pad(tgt_mask,[0,1])
            self.model.inference_iter()
            tgt, past_tgts = self.model.inference_step(tgt=tgt, 
                                                    pep_mass=pep_mass,
                                                    tgt_mask=tgt_mask,
                                                    past_tgts=past_tgts,
                                                    past_mems=past_mems,
                                                    mem_key_padding_mask=mem_key_padding_mask)
            tgt = tgt.float().cpu()
            pep_status_list, pep_finish_pool, edges_info, keep_index = self.pep_status_update(pep_status_list, edges_info, pep_finish_pool, tgt, knapsack_mask, precursor_mass)
            
            past_tgts = [(k[keep_index], v[keep_index]) for k,v in past_tgts]
            tgt_mask = tgt_mask[keep_index]
            past_mems = [(k[keep_index], v[keep_index]) for k,v in past_mems]
            mem_key_padding_mask = mem_key_padding_mask[keep_index]
            precursor_mass = precursor_mass[keep_index]
            edge_mass_threshold = edge_mass_threshold[keep_index]
            pep_mass_threshold = pep_mass_threshold[keep_index]

        return pep_finish_pool

    def pep_status_initializer(self, edges_info, idx):
        pep_status_list = []
        for i, edges in zip(idx, edges_info):
            pep_status_list.append(Pep_Inference_Status(idx=i,
                                                        inference_seq = ['<answer>'],
                                                        current_mass = edges[0].edge_start_mass,
                                                        current_end_mass = edges[0].edge_end_mass,
                                                        current_pepblock_index = edges[0].edge_index))
        return pep_status_list

    def model_input_generator(self, pep_status_list, precursor_mass, pep_mass_threshold, edge_mass_threshold):
        tgt = []
        pepblock_index = []
        aa_index_in_block = []

        current_aa_mass = []
        current_edge_mass = []

        for pep_status in pep_status_list:
            tgt.append(self.aa_dict[pep_status.inference_seq[-1]])
            pepblock_index.append(pep_status.current_pepblock_index)
            aa_index_in_block.append(pep_status.current_aaindex)
            current_aa_mass.append(pep_status.current_mass)
            current_edge_mass.append(pep_status.current_end_mass)
        
        current_aa_mass = np.array(current_aa_mass)
        current_edge_mass = np.array(current_edge_mass)
        current_pep_remain_mass = precursor_mass-current_aa_mass
        current_edge_remain_mass = current_edge_mass-current_aa_mass
        knapsack_mask = self.knapsack_mask_builder(current_pep_remain_mass, current_edge_remain_mass, pep_mass_threshold, edge_mass_threshold)
        pep_mass = self.peptide_mass_embedding(current_aa_mass, current_pep_remain_mass)
        
        tgt = self.to_cuda({'tgt':torch.LongTensor(tgt).unsqueeze(1),'pepblock_index':torch.LongTensor(pepblock_index).unsqueeze(1),'aa_index_in_block':torch.LongTensor(aa_index_in_block).unsqueeze(1)})
        pep_mass = self.to_cuda(pep_mass)
        return tgt, pep_mass, knapsack_mask
    
    def knapsack_mask_builder(self, current_pep_remain_mass, current_edge_remain_mass, pep_mass_threshold, edge_mass_threshold):
        knapsack_mask = []
        current_pep_remain_lowerbound = current_pep_remain_mass - pep_mass_threshold
        current_pep_remain_upperbound = current_pep_remain_mass + pep_mass_threshold

        current_edge_remain_lowerbound = current_edge_remain_mass - edge_mass_threshold
        current_edge_remain_upperbound = current_edge_remain_mass + edge_mass_threshold

        current_edge_remain_lowerbound_3000 = self.knapsack_edge_mass.searchsorted(current_edge_remain_lowerbound)
        current_edge_remain_upperbound_3000 = self.knapsack_edge_mass.searchsorted(current_edge_remain_upperbound)
        
        current_pep_remain_lowerbound = self.knapsack_mass.searchsorted(current_pep_remain_lowerbound)
        current_pep_remain_upperbound = self.knapsack_mass.searchsorted(current_pep_remain_upperbound)

        current_edge_remain_lowerbound = self.knapsack_mass.searchsorted(current_edge_remain_lowerbound)
        current_edge_remain_upperbound = self.knapsack_mass.searchsorted(current_edge_remain_upperbound)

        for pep_lowerbound, pep_upperbound, edge_lowerbound, edge_upperbound, edge_lowerbound_3000, edge_upperbound_3000 in zip(current_pep_remain_lowerbound,current_pep_remain_upperbound,current_edge_remain_lowerbound,current_edge_remain_upperbound,current_edge_remain_lowerbound_3000,current_edge_remain_upperbound_3000):
            if edge_upperbound_3000<self.knapsack_edge_mass_len:
                candidate_aa = set(''.join(self.knapsack_edge_aa_composition[edge_lowerbound_3000:edge_upperbound_3000]))
                temp = torch.ones(20, dtype=bool)
                temp[[self.tokenize_aa_dict[aa] for aa in candidate_aa]] = 0
            else:
                temp = torch.zeros(20, dtype=bool)
            '''if pep_upperbound<self.knapsack_mass_len and edge_upperbound<self.knapsack_mass_len:
                candidate_aa_pep = self.knapsack_aa_composition[self.knapsack_index[pep_lowerbound]:self.knapsack_index[pep_upperbound]]
                candidate_aa_edge = self.knapsack_aa_composition[self.knapsack_index[edge_lowerbound]:self.knapsack_index[edge_upperbound]]
                if pep_upperbound<=edge_upperbound:
                    temp = torch.ones(20, dtype=bool)
                    candidate_aa = candidate_aa_pep.any(0)
                    if candidate_aa.any():
                        temp[candidate_aa] = 0
                    else:
                        candidate_aa = candidate_aa_edge.any(0)
                        temp[candidate_aa] = 0
                else:
                if pep_upperbound>edge_upperbound and candidate_aa_pep.shape[0]<2000 and candidate_aa_edge.shape[0]<2000:
                    candidate_aa_pep = candidate_aa_pep[:,np.newaxis,:]
                    candidate_aa_edge = candidate_aa_edge[np.newaxis,...]
                    temp = torch.ones(20, dtype=bool)
                    candidate_aa = candidate_aa_edge[0][(candidate_aa_edge<=candidate_aa_pep).all(-1).any(0)].any(0)
                    if candidate_aa.any():
                        temp[candidate_aa] = 0
                    else:
                        candidate_aa = candidate_aa_edge[0].any(0)
                        temp[candidate_aa] = 0
                else:
                    temp = torch.ones(20, dtype=bool)
                    candidate_aa = candidate_aa_edge.any(0)
                    temp[candidate_aa] = 0
            elif pep_upperbound==self.knapsack_mass_len and edge_upperbound<self.knapsack_mass_len:
                candidate_aa_edge = candidate_aa_edge
                temp = torch.ones(20, dtype=bool)
                candidate_aa = candidate_aa_edge.any(0)
                temp[candidate_aa] = 0
            elif pep_upperbound==self.knapsack_mass_len and edge_upperbound==self.knapsack_mass_len and edge_upperbound_3000<self.knapsack_edge_mass_len:
                candidate_aa = set(''.join(self.knapsack_edge_aa_composition[edge_lowerbound_3000:edge_upperbound_3000]))
                temp = torch.ones(20, dtype=bool)
                temp[[self.tokenize_aa_dict[aa] for aa in candidate_aa]] = 0
            else:
                temp = torch.zeros(20, dtype=bool)'''
            knapsack_mask.append(temp)
        knapsack_mask = torch.stack(knapsack_mask)
        return knapsack_mask

    def peptide_mass_embedding(self, nterm_mass, cterm_mass):
        nterm_mass_1_sin, nterm_mass_1_cos = self.sinusoidal_position_embedding(nterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_1_sin, cterm_mass_1_cos = self.sinusoidal_position_embedding(cterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        nterm_mass_2_sin, nterm_mass_2_cos = self.sinusoidal_position_embedding(nterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_2_sin, cterm_mass_2_cos = self.sinusoidal_position_embedding(cterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        pe_sin = torch.stack([nterm_mass_1_sin,cterm_mass_1_sin,nterm_mass_2_sin,cterm_mass_2_sin],dim=1).repeat(1,self.nhead//4,1).unsqueeze(1)
        pe_cos = torch.stack([nterm_mass_1_cos,cterm_mass_1_cos,nterm_mass_2_cos,cterm_mass_2_cos],dim=1).repeat(1,self.nhead//4,1).unsqueeze(1)
        return pe_sin, pe_cos

    def pep_status_update(self, pep_status_list, edges_info, pep_finish_pool, tgt, knapsack_mask, precursors_mass):
        keep_index = np.ones(len(pep_status_list), dtype=bool)
        for i, (current_status, edge_info, precursor_mass) in enumerate(zip(pep_status_list, edges_info, precursors_mass)):
            if abs(current_status.current_end_mass-current_status.current_mass)<precursor_mass*self.ms1_threshold_ppm*1e-6+self.ms2_threshold_da*2:
                current_status.current_pepblock_count += 1
                if current_status.current_pepblock_count >= len(edge_info):
                    label_seq = self.psm_head.iloc[current_status.idx]['Annotated Sequence']
                    inference_seq = ''.join(current_status.inference_seq[1:]).split('<answer>')
                    train_seq = []
                    pepblock_count = 0
                    for pepblock in label_seq.split(' '):
                        if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                            train_seq += [inference_seq[pepblock_count]]
                            pepblock_count += 1
                        else: train_seq += [pepblock]
                    train_seq = ' '.join(train_seq)
                    pep_finish_pool += [Pep_Finish_Status(idx=current_status.idx, train_seq=train_seq, max_loss=current_status.max_loss)]
                    keep_index[i] = False
                else:
                    current_status.edge_true = True
                    current_status.inference_seq += ['<answer>']
                    current_status.current_aaindex = 0
                    current_status.current_pepblock_index = edge_info[current_status.current_pepblock_count].edge_index
                    current_status.current_mass = edge_info[current_status.current_pepblock_count].edge_start_mass
                    current_status.current_end_mass = edge_info[current_status.current_pepblock_count].edge_end_mass
            else:
                result = tgt[i,0,:20].masked_fill(knapsack_mask[i], -float('inf')).softmax(-1).numpy()
                if result[result.argmax()]<0.7:
                    max_aa_idx = result.argmax()
                    result[max_aa_idx] = 0
                    result = result/result.sum()*0.3
                    result[max_aa_idx] = 0.7
                elif 0.95<result[result.argmax()]<1:
                    max_aa_idx = result.argmax()
                    result[max_aa_idx] = 0
                    result = result/result.sum()*0.05
                    result[max_aa_idx] = 0.95
                inference_aa = self.detokenize_aa_dict[np.random.choice(20, p=result)]

                if current_status.edge_true:
                    label_aa = edge_info[current_status.current_pepblock_count].edge_seq[current_status.current_aaindex]
                    label = torch.zeros(21)
                    label[self.tokenize_aa_dict[label_aa]] = 1
                    if current_status.pep_true: label[-1] = 1
                    if label_aa!=inference_aa:
                        current_status.edge_true = False
                        current_status.pep_true = False
                else:
                    label = torch.zeros(21)
                
                loss = self.loss(tgt[i,0],label).item()
                if current_status.max_loss < loss: current_status.max_loss = loss
                current_status.inference_seq+=[inference_aa]
                current_status.current_aaindex+=1
                current_status.current_mass+=Residual_seq(inference_aa).mass
        
        new_pep_status_list = []
        new_edges_info = []
        for keep_flag, current_status, edge_info in zip(keep_index, pep_status_list, edges_info):
            if keep_flag:
                new_pep_status_list.append(current_status)
                new_edges_info.append(edge_info)

        return new_pep_status_list, pep_finish_pool, new_edges_info, keep_index

    def edge_info(self, seq):
        tgt_seq = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                tgt_seq += [pepblock]
                
        pepblock_index = []
        for i, pepblock in enumerate(seq.split(' '),start=1):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                pepblock_index += [i]

        mass = 0
        edge_start_mass = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                edge_start_mass += [mass]
                mass += Residual_seq(pepblock).mass
            else:
                mass += Residual_seq(pepblock).mass

        mass = 0
        edge_end_mass = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                mass += Residual_seq(pepblock).mass
                edge_end_mass += [mass]
            else:
                mass += Residual_seq(pepblock).mass
        
        return [Edge_Info(edge_seq=tgt_seq[i], edge_index=pepblock_index[i], edge_start_mass=edge_start_mass[i], edge_end_mass=edge_end_mass[i]) for i in range(len(tgt_seq))]

    @staticmethod
    def sinusoidal_position_embedding(mass_position, dim, lambda_max, lambda_min):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*math.pi)
        scale = lambda_min/lambda_max
        div_term = base*scale**(np.arange(0, dim, 2)/dim)
        pe_sin = np.sin(mass_position[...,np.newaxis] / div_term)
        pe_cos = np.cos(mass_position[...,np.newaxis] / div_term)
        return torch.Tensor(pe_sin), torch.Tensor(pe_cos)

    def to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.local_rank, non_blocking=True)
        elif isinstance(data, string_classes):
            return data
        elif isinstance(data, collections.abc.Mapping):
            try:
                return type(data)({k: self.to_cuda(sample) for k, sample in data.items()})  # type: ignore[call-arg]
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {k: self.to_cuda(sample) for k, sample in data.items()}
        elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
            return type(data)(*(self.to_cuda(sample) for sample in data))
        elif isinstance(data, tuple):
            return [self.to_cuda(sample) for sample in data]  # Backwards compatibility.
        elif isinstance(data, collections.abc.Sequence):
            try:
                return type(data)([self.to_cuda(sample) for sample in data])  # type: ignore[call-arg]
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [self.to_cuda(sample) for sample in data]
        elif hasattr(data, "to"):
            return data.to(device=self.local_rank, non_blocking=True)
        else:
            return data

class EnvironmentDataset(Dataset):
    def __init__(self, aa_dict, data, psm_head, max_len=32, hidden_size=768, max_point=256, nhead=12):
        super().__init__()
        self.aa_dict = aa_dict
        self.data = data
        self.psm_head = psm_head
        self.single_aa_mask_list = ['N','Q','K','m','F','R','W']
        self.max_len = max_len+1
        self.hidden_size = hidden_size
        self.max_point = max_point
        self.nhead = nhead
    
    def __getitem__(self, idx):
        seq,moverz,charge,experiment,file_id,scan = self.psm_head.iloc[idx]
        
        precursor_mass = Ion.precursorion2mass(moverz,charge)

        ion_moverz_encoder, ion_moverz_decoder, intensity, ion_mask_encoder, ion_mask_decoder = self.spectra_processor(experiment,file_id,scan)
        tgt_mask = self.decoder_mask(seq)
        tgt = self.tonkenize(seq)
        pepblock_index, aa_index_in_block = self.token_position(seq)
        peptide_mass_embedding = self.peptide_mass_embedding(seq, precursor_mass)
        return [{'src':intensity,
                 'charge':charge},
                {'src_key_padding_mask':ion_mask_encoder,
                 'peaks_moverz':ion_moverz_encoder},
                {'tgt':tgt,
                 'pepblock_index':pepblock_index, 
                 'aa_index_in_block':aa_index_in_block},
                tgt_mask,
                peptide_mass_embedding,
                ion_mask_decoder,
                ion_moverz_decoder,
                idx]

        
    def __len__(self):
        return len(self.psm_head)

    def spectra_processor(self,experiment,file_id,scan):
        ion_moverz, intensity = self.data[':'.join([experiment,file_id,str(scan)])]
        spectra_mask_encoder = torch.zeros([1,self.max_point],dtype=bool)
        spectra_mask_encoder[...,len(ion_moverz):] = True
        spectra_mask_decoder = spectra_mask_encoder.unsqueeze(0)
        ion_moverz = np.pad(ion_moverz,(1,self.max_point-len(ion_moverz)-1))
        intensity = np.pad(intensity,(1,self.max_point-len(intensity)-1))
        intensity = intensity/intensity.max()
        ion_moverz_encoder = self.sinusoidal_position_embedding(ion_moverz, 128, 1e4, 1e-5)
        ion_moverz_decoder_sin, ion_moverz_decoder_cos = self.sinusoidal_position_embedding(ion_moverz, 64, 1e4, 1e-5)
        ion_moverz_decoder = ion_moverz_decoder_sin.unsqueeze(1), ion_moverz_decoder_cos.unsqueeze(1)
        intensity_sin, intensity_cos = self.sinusoidal_position_embedding(intensity, self.hidden_size, 1, 1e-7)
        intensity = torch.concat([intensity_sin,intensity_cos],dim=-1)
        return ion_moverz_encoder, ion_moverz_decoder, intensity, spectra_mask_encoder, spectra_mask_decoder

    def tonkenize(self, seq):
        fix_seq = [self.aa_dict['<bos>']]
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                fix_seq += [self.aa_dict['<x>']]
            else: fix_seq += [self.aa_dict[pepblock]]
        tgt = torch.LongTensor(fix_seq)
        tgt = pad(tgt,[0,self.max_len-tgt.shape[0]])
        return tgt
    
    def decoder_mask(self, seq):
        fix_len = len(seq.split(' '))+1
        tgt_mask = torch.ones(1,1,self.max_len,dtype=bool)
        tgt_mask[0,0,:fix_len] = 0
        return tgt_mask
    
    def token_position(self, seq):
        pepblock_index = [i for i in range(1,len(seq.split(' '))+1)]
        aa_index_in_block = [0]*len(seq.split(' '))
        pepblock_index = torch.LongTensor([0]+pepblock_index)
        aa_index_in_block = torch.LongTensor([0]+aa_index_in_block)
        pepblock_index = pad(pepblock_index,[0,self.max_len-len(pepblock_index)])
        aa_index_in_block = pad(aa_index_in_block,[0,self.max_len-len(aa_index_in_block)])
        return pepblock_index, aa_index_in_block
    
    def peptide_mass_embedding(self, seq, precursor_mass):
        mass = 0
        nterm_mass_fix = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                mass += Residual_seq(pepblock).mass
                #nterm_mass_fix += [0]
                nterm_mass_fix += [mass]
            else:
                mass += Residual_seq(pepblock).mass
                nterm_mass_fix += [mass]
        nterm_mass = [0]+nterm_mass_fix
        cterm_mass = precursor_mass - nterm_mass
        nterm_mass = np.pad(nterm_mass,(0,self.max_len-len(nterm_mass)))
        cterm_mass = np.pad(cterm_mass,(0,self.max_len-len(cterm_mass)))
        nterm_mass_1_sin, nterm_mass_1_cos = self.sinusoidal_position_embedding(nterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_1_sin, cterm_mass_1_cos = self.sinusoidal_position_embedding(cterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        nterm_mass_2_sin, nterm_mass_2_cos = self.sinusoidal_position_embedding(nterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_2_sin, cterm_mass_2_cos = self.sinusoidal_position_embedding(cterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        pe_sin = torch.stack([nterm_mass_1_sin,cterm_mass_1_sin,nterm_mass_2_sin,cterm_mass_2_sin],dim=1).repeat(1,self.nhead//4,1)
        pe_cos = torch.stack([nterm_mass_1_cos,cterm_mass_1_cos,nterm_mass_2_cos,cterm_mass_2_cos],dim=1).repeat(1,self.nhead//4,1)
        return pe_sin, pe_cos

    @staticmethod
    def sinusoidal_position_embedding(mass_position, dim, lambda_max, lambda_min):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*math.pi)
        scale = lambda_min/lambda_max
        div_term = base*scale**(np.arange(0, dim, 2)/dim)
        pe_sin = np.sin(mass_position[...,np.newaxis] / div_term)
        pe_cos = np.cos(mass_position[...,np.newaxis] / div_term)
        return torch.Tensor(pe_sin), torch.Tensor(pe_cos)