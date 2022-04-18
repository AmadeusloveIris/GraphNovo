import os
import gzip
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from genova.utils.BasicClass import Residual_seq

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, spec_header, dataset_dir_path, aa_datablock_dict = None):
        super().__init__()
        self.cfg = cfg
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path
        assert aa_datablock_dict or cfg.task == 'node_classification'
        self.aa_datablock_dict = aa_datablock_dict

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.loc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))

        spec['node_input']['charge'] = spec_head['Charge']
        graph_label = spec.pop('graph_label').T
        graph_label = graph_label[graph_label.any(-1)]
        node_mass = spec.pop('node_mass')
        seq = spec_head['Annotated Sequence']
        if self.cfg.task == 'node_classification':
            spec['graph_label'] = torch.any(graph_label, 0).long()
            return spec
        
        elif self.cfg.task == 'optimum_path_sequence':
            raise NotImplementedError
            
        elif self.cfg.task == 'sequence_generation':
            target = {}
            seq_blocks = self.seq2seqblock(seq, graph_label)
            target['tgt'] = seq_blocks
            target['trans_mask'] = self.trans_mask_sequence_generation(seq_blocks, node_mass)
            return spec, target
        
        elif self.cfg.task == 'optimum_path':
            trans_mask = self.trans_mask_optimum_path(node_mass, graph_label, spec['rel_input']['dist'])
            graph_probability = torch.Tensor(self.graph_probability_gen(graph_label))
            tgt = {}
            tgt['tgt'] = graph_probability[:-1]
            tgt['trans_mask'] = trans_mask
            return spec, tgt, graph_probability[1:]
            
    def graph_probability_gen(self, graph_label):
        graph_probability = graph_label/graph_label.sum(-1).unsqueeze(1)
        return graph_probability
    
    def trans_mask_sequence_generation(self,seq_blocks,node_mass):
        seq_mass = np.array([Residual_seq(seq_block.replace('L','I')).mass for seq_block in seq_blocks]).cumsum()
        trans_mask = torch.zeros((seq_mass.size,node_mass.size))
        trans_mask[0,0] = -float('inf')
        for i, board in enumerate(node_mass.searchsorted(seq_mass+0.02,side='right')[:-1],start=1):
            trans_mask[i,:board] = -float('inf')
        return trans_mask
    
    def trans_mask_optimum_path(self, node_mass, graph_label, dist):
        node_num = node_mass.size
        edge_mask = torch.zeros(node_num,node_num,dtype=bool)
        for x,y in enumerate(node_mass.searchsorted(node_mass+max(self.aa_datablock_dict.values())+0.04)):
            edge_mask[x,y:] = True
        edge_mask = torch.logical_or(edge_mask,dist!=0)
        trans_mask=((graph_label@edge_mask.int())!=0).bool()
        trans_mask = torch.where(trans_mask,0.0,-float('inf'))
        return trans_mask[:-1]
    
    def seq2seqblock(self, seq, graph_label):
        seq_block = []
        for i, combine_flag in enumerate(~graph_label.any(-1)[1:]):
            if combine_flag:
                if 'combine_start_index' not in locals():
                    combine_start_index = i
            else:
                try:
                    if i+1-combine_start_index>6: seq_block.append('X')
                    else:
                        seq_block.append(seq[combine_start_index:i+1])
                        del(combine_start_index)
                except:
                    seq_block.append(seq[i])
        return seq_block
    
    def __len__(self):
        return len(self.spec_header)