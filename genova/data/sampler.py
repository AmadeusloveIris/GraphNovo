import torch
import numpy as np
from random import choices
from torch.utils.data import Sampler


class GenovaBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, cfg, device, gpu_capacity_scaller, spec_header, bin_boarders, shuffle=True) -> None:
        super().__init__(data_source=None)
        self.cfg = cfg
        self.bin_boarders = bin_boarders
        self.gpu_capacity = torch.cuda.get_device_properties(device).total_memory*gpu_capacity_scaller
        self.shuffle = shuffle
        self.spec_header = spec_header
        
        self.hidden_size = self.cfg['hidden_size']
        self.d_relation = self.cfg['encoder']['d_relation']
        self.num_layers = self.cfg['encoder']['num_layers']
        self.d_node = self.cfg['encoder']['node_encoder']['d_node']
        self.d_node_expansion = self.cfg['encoder']['node_encoder']['expansion_factor']
        self.edge_expansion = self.cfg['encoder']['edge_encoder']['expansion_factor']
        self.edge_d_edge = self.cfg['encoder']['edge_encoder']['d_edge']
        self.path_expansion = self.cfg['encoder']['path_encoder']['expansion_factor']
        self.path_d_edge = self.cfg['encoder']['path_encoder']['d_edge']

        self.node_sparse = 4 * ((29 + self.d_node_expansion) * self.d_node)
        self.node = 4 * ((2 * self.d_node_expansion)*self.d_node + 4 * (self.d_node_expansion*self.d_node+self.hidden_size)/2)
        
        self.relation_matrix = 4 * 7 * self.d_relation * self.num_layers
        self.relation_ffn = 4 * (3 * self.d_relation + 13 * self.hidden_size) * self.num_layers + 4*8*self.hidden_size
        
        self.edge_matrix = 4 * (8*self.edge_d_edge + 2*self.d_relation + 2 * self.edge_expansion*self.edge_d_edge)
        self.edge_sparse = 4 * (4 + self.edge_expansion) * self.edge_d_edge
        
        self.path_matrix = 4 * (8*self.path_d_edge + 2*self.d_relation + 4*self.path_expansion*self.path_d_edge)
        self.path_sparse = 4 * (9 + self.path_expansion) * self.path_d_edge

    def __iter__(self):
        if self.shuffle: self.spec_header = self.spec_header.sample(frac=1)
        self.generate_bins()
        return self

    def __next__(self):
        if self.bins_readpointer.sum()==len(self.spec_header): raise StopIteration
        bin_index = choices([i for i in range(self.bin_len.size)], \
                            weights=self.bin_len-self.bins_readpointer)[0]
        bin = self.bins[bin_index]
        max_node = 0
        edge_num = 0
        path_num = 0
        for i in range(self.bins_readpointer[bin_index], len(bin)):
            spec_index=bin.iloc[i]
            if spec_index['Node Number']>max_node: max_node = spec_index['Node Number']
            batch_num = i-self.bins_readpointer[bin_index]+1
            edge_num += spec_index['Edge Num']
            path_num += spec_index['Relation Num']
            node_consumer = self.node_sparse*max_node*batch_num*30 + self.node*max_node*batch_num
            edge_consumer = self.edge_matrix*max_node**2*batch_num + self.edge_sparse*edge_num
            path_consumer = self.path_matrix*max_node**2*batch_num + self.path_sparse*path_num
            relation_cosumer = self.relation_matrix*max_node**2*batch_num + self.relation_ffn*max_node*batch_num
            theo = node_consumer+edge_consumer+path_consumer+relation_cosumer
            if theo>self.gpu_capacity:
                index = bin.iloc[self.bins_readpointer[bin_index]:i].index
                self.bins_readpointer[bin_index] = i
                return index
        index = bin.iloc[self.bins_readpointer[bin_index]:].index
        self.bins_readpointer[bin_index] = len(bin)
        return index
        
    def __len__(self):
        return len(self.spec_header)

    def generate_bins(self):
        if self.shuffle: self.spec_header.sample(frac=1)
        self.bins = [self.spec_header[np.logical_and(self.spec_header['Node Number']>self.bin_boarders[i], \
            self.spec_header['Node Number']<=self.bin_boarders[i+1])] for i in range(len(self.bin_boarders)-1)]
        self.bin_len = np.array([len(bin_index) for bin_index in self.bins])
        self.bins_readpointer = np.zeros(len(self.bin_boarders)-1,dtype=int)