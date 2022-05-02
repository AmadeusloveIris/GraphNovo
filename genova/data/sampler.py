import torch
import numpy as np
from typing import List, Union
from random import choices
import torch.distributed as dist
from dataclasses import dataclass
from torch.utils.data import Sampler

@dataclass
class EncoderMem:
    def __init__(self,cfg):
        self.cfg = cfg
        self.hidden_size = self.cfg['encoder']['hidden_size']
        self.d_relation = self.cfg['encoder']['d_relation']
        self.num_layers = self.cfg['encoder']['num_layers']
        self.d_node = self.cfg['encoder']['node_encoder']['d_node']
        self.d_node_expansion = self.cfg['encoder']['node_encoder']['expansion_factor']
        self.edge_expansion = self.cfg['encoder']['edge_encoder']['expansion_factor']
        self.edge_d_edge = self.cfg['encoder']['edge_encoder']['d_edge']
        self.path_expansion = self.cfg['encoder']['path_encoder']['expansion_factor']
        self.path_d_edge = self.cfg['encoder']['path_encoder']['d_edge']

        # Encoding Node 所需显存消耗
        self.node_sparse = 4 * ((29 + self.d_node_expansion) * self.d_node)
        self.node = 4 * ((3 * self.d_node_expansion) * self.d_node + 4 * (self.d_node_expansion * self.d_node + self.hidden_size)/2)

        # Direct Edge Graph 所需显存消耗        
        self.edge_matrix = 4 * (3*self.d_relation + 2*self.edge_expansion*self.edge_d_edge)
        self.edge_sparse = 4 * (4 + self.edge_expansion) * self.edge_d_edge

        # Longest Path Graph 所需显存消耗
        self.path_matrix = 4 * (3*self.d_relation + 4*self.path_expansion*self.path_d_edge)
        self.path_sparse = 4 * (9 + self.path_expansion) * self.path_d_edge

        # Encoder Layer 所需显存消耗
        self.relation_matrix = 4 * 8 * self.d_relation * self.num_layers
        self.relation_linear = 4 * (2 * self.d_relation + 4 * self.hidden_size) * self.num_layers
        self.relation_ffn = 4 * 22 * self.hidden_size * self.num_layers

@dataclass
class DecoderMem:
    def __init__(self,cfg):
        self.cfg = cfg
        self.hidden_size = self.cfg['decoder']['hidden_size']
        self.d_relation = self.cfg['decoder']['d_relation']
        self.num_layers = self.cfg['decoder']['num_layers']

        # Encoder Layer 所需显存消耗
        self.self_relation_matrix = 4 * 4 * self.d_relation * self.num_layers
        self.self_relation_linear = 4 * (2 * self.d_relation + 4 * self.hidden_size) * self.num_layers

        self.trans_relation_matrix = 4 * 4 * self.d_relation * self.num_layers
        self.trans_relation_linear = 4 * (2 * self.d_relation + 4 * self.hidden_size) * self.num_layers

        self.relation_ffn = 4 * 22 * self.hidden_size * self.num_layers

class GenovaBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, cfg, device: Union[int, torch.device], 
                 gpu_capacity_scaller: float, spec_header, 
                 bin_boarders: List, model, shuffle=True, 
                 sample_time_limitation = 20000) -> Sampler:
        super().__init__(data_source=None)
        self.cfg = cfg
        self.bin_boarders = np.array(bin_boarders)
        self.gpu_capacity = torch.cuda.get_device_properties(device).total_memory*gpu_capacity_scaller
        self.shuffle = shuffle
        self.spec_header_ori = spec_header
        self.model_mem = sum([param.nelement() for param in model.parameters()])*4*4
        self.encoder_mem = EncoderMem(cfg)
        if 'decoder' in cfg: self.decoder_mem = DecoderMem(cfg)
        if shuffle: self.t_bzs_proportion = self.bzs_sampling(sample_time_limitation)

    def __iter__(self):
        self.generate_bins()
        return self

    def __next__(self):
        if (self.bins_readpointer+1).sum()>=self.bin_len.sum(): raise StopIteration
        # 警告： 由于分bucket，并且每个bucket的batch size不同，所以不能直接以每个bucket中剩余的数据量做为权重，
        # 需要考虑个bucket大致的batch size的比例，并加以修正。否则将导致抽取不均衡，平均来看，batch size大的bucket
        # 将会被先抽。
        if self.shuffle:
            bin_index = choices([i for i in range(self.bin_len.size)], \
                                weights=(self.bin_len-(self.bins_readpointer+1))/self.t_bzs_proportion)[0]
        else:
            bin_index = choices([i for i in range(self.bin_len.size)], \
                                weights=(self.bin_len-(self.bins_readpointer+1)))[0]

        bin = self.bins[bin_index]
        max_node = 0
        edge_num = 0
        path_num = 0
        i = self.bins_readpointer[bin_index]
        while i<len(bin):
            spec_index=bin.iloc[i]
            if spec_index['Node Number']>max_node: max_node = spec_index['Node Number']
            batch_num = i-self.bins_readpointer[bin_index]+1
            edge_num += spec_index['Edge Num']
            path_num += spec_index['Relation Num']
            
            encoder_node_consumer = self.encoder_mem.node_sparse * max_node * batch_num * 30 + self.encoder_mem.node * max_node * batch_num
            encoder_edge_consumer = self.encoder_mem.edge_matrix * max_node**2 * batch_num + self.encoder_mem.edge_sparse * edge_num
            encoder_path_consumer = self.encoder_mem.path_matrix * max_node**2 * batch_num + self.encoder_mem.path_sparse * path_num
            encoder_relation_cosumer = self.encoder_mem.relation_matrix * max_node**2 * batch_num + self.encoder_mem.relation_ffn * max_node * batch_num
            encoder_theo = encoder_node_consumer + encoder_edge_consumer + encoder_path_consumer + encoder_relation_cosumer
            if self.cfg.task != 'node_classification':
                decoder_self_relation = self.decoder_mem.self_relation_matrix * 32**2 * batch_num
                decoder_trans_relation = self.decoder_mem.self_relation_matrix * 32*max_node * batch_num
                decoder_ffn = (self.decoder_mem.self_relation_linear+self.decoder_mem.trans_relation_linear+self.decoder_mem.relation_ffn) * 32 * batch_num
                decoder_theo = decoder_self_relation+decoder_trans_relation+decoder_ffn
                theo = encoder_theo + decoder_theo + self.model_mem
            else:
                theo = encoder_theo + self.model_mem
            
            if theo>self.gpu_capacity:
                candidate_batch_size = i-self.bins_readpointer[bin_index]
                if candidate_batch_size==0: #如果显卡内存容量一条数据都放不了就直接跳过
                    i += 1
                    self.bins_readpointer[bin_index] = i
                    max_node = 0
                    edge_num = 0
                    path_num = 0
                    continue
                else:
                    if candidate_batch_size//8 > 0: i = i-candidate_batch_size%8 #强制batch size可以被8整除，优化计算效率(虽然我测试过没什么用)
                    index = bin.iloc[self.bins_readpointer[bin_index]:i].index
                    self.bins_readpointer[bin_index] = i
                    return index
            else: 
                i += 1
        index = bin.iloc[self.bins_readpointer[bin_index]:].index
        self.bins_readpointer[bin_index] = len(bin)-1
        return index
        
    def __len__(self):
        return len(self.spec_header)

    def generate_bins(self):
        if self.shuffle: self.spec_header_ori = self.spec_header_ori.sample(frac=1,random_state=0)
        if dist.is_initialized(): self.spec_header = self.spec_header_ori.iloc[dist.get_rank()::dist.get_world_size()] # subset of dataset for ddp
        else: self.spec_header = self.spec_header_ori
        self.bins = [self.spec_header[np.logical_and(self.spec_header['Node Number']>self.bin_boarders[i], \
            self.spec_header['Node Number']<=self.bin_boarders[i+1])] for i in range(len(self.bin_boarders)-1)]
        self.bin_len = np.array([len(bin_index) for bin_index in self.bins])
        self.bins_readpointer = np.zeros(len(self.bin_boarders)-1,dtype=int)
    
    def bzs_sampling(self,sample_time_limitation):
        spec_header = self.spec_header_ori.sample(frac=1)
        bins = [spec_header[np.logical_and(spec_header['Node Number']>self.bin_boarders[i], \
            spec_header['Node Number']<=self.bin_boarders[i+1])] for i in range(len(self.bin_boarders)-1)]
        bins_readpointer = np.zeros(len(self.bin_boarders)-1,dtype=int)

        sample_num = np.zeros(len(bins))
        sample_count = np.zeros(len(bins))

        for bin_index in range(len(bins)):
            bin = bins[bin_index]
            i = bins_readpointer[bin_index]
            sample_time = 0
            max_node = 0
            edge_num = 0
            path_num = 0
            while sample_time<sample_time_limitation:
                if i >= len(bin): break
                sample_time += 1
                spec_index=bin.iloc[i]
                if spec_index['Node Number']>max_node: max_node = spec_index['Node Number']
                batch_num = i-bins_readpointer[bin_index]+1
                edge_num += spec_index['Edge Num']
                path_num += spec_index['Relation Num']
                
                encoder_node_consumer = self.encoder_mem.node_sparse * max_node * batch_num * 30 + self.encoder_mem.node * max_node * batch_num
                encoder_edge_consumer = self.encoder_mem.edge_matrix * max_node**2 * batch_num + self.encoder_mem.edge_sparse * edge_num
                encoder_path_consumer = self.encoder_mem.path_matrix * max_node**2 * batch_num + self.encoder_mem.path_sparse * path_num
                encoder_relation_cosumer = self.encoder_mem.relation_matrix * max_node**2 * batch_num + self.encoder_mem.relation_ffn * max_node * batch_num
                encoder_theo = encoder_node_consumer + encoder_edge_consumer + encoder_path_consumer + encoder_relation_cosumer
                if 'decoder' in self.cfg:
                    decoder_self_relation = self.decoder_mem.self_relation_matrix * 32**2 * batch_num
                    decoder_trans_relation = self.decoder_mem.self_relation_matrix * 32*max_node * batch_num
                    decoder_ffn = (self.decoder_mem.self_relation_linear+self.decoder_mem.trans_relation_linear+self.decoder_mem.relation_ffn) * 32 * batch_num
                    decoder_theo = decoder_self_relation+decoder_trans_relation+decoder_ffn
                    theo = encoder_theo + decoder_theo + self.model_mem
                else:
                    theo = encoder_theo + self.model_mem
                if theo>self.gpu_capacity:
                    candidate_batch_size = i-bins_readpointer[bin_index]
                    if candidate_batch_size==0: #如果显卡内存容量一条数据都放不了就直接跳过
                        i += 1
                        bins_readpointer[bin_index] = i
                        max_node = 0
                        edge_num = 0
                        path_num = 0
                        continue
                    else:
                        if candidate_batch_size//8 > 0: i = i-candidate_batch_size%8 #强制batch size可以被8整除，优化计算效率(虽然我测试过没什么用)
                        index = bin.iloc[bins_readpointer[bin_index]:i].index
                        bins_readpointer[bin_index] = i
                        sample_num[bin_index] += len(index)
                        sample_count[bin_index] += 1
                        max_node = 0
                        edge_num = 0
                        path_num = 0
                else: 
                    i += 1
        return sample_num/sample_count
