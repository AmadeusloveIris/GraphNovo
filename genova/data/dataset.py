import os
import gzip
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from genova.utils.BasicClass import Residual_seq

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path
        self.aa_mass_dict = {aa:Residual_seq(aa).mass for aa in Residual_seq.output_aalist()}
        self.aa_id = {aa:i for i, aa in enumerate(Residual_seq.output_aalist(),start=3)}
        self.aa_id['<pad>'] = 0
        self.aa_id['<bos>'] = 1
        self.aa_id['<eos>'] = 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.loc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))

        spec['node_input']['charge'] = spec_head['Charge']
        spec['node_input']['rt'] = spec_head['iRT']
        graph_label = spec.pop('graph_label').T
        graph_label = graph_label[graph_label.any(-1)]
        node_mass = spec.pop('node_mass')
        seq = spec_head['Annotated Sequence'].replace('L','I')
        if self.cfg.task == 'node_classification':
            graph_label = torch.any(graph_label, 0).float()
            return spec, graph_label
        
        elif self.cfg.task == 'optimum_path_sequence':
            raise NotImplementedError
            
        elif self.cfg.task == 'sequence_generation':
            tgt = {}
            seq_id = self.seq2id(seq)
            tgt['tgt'] = seq_id[:-1]
            tgt['trans_mask'] = self.trans_mask_sequence_generation(seq, node_mass)
            return spec, tgt, seq_id[1:]
        
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
    
    def trans_mask_sequence_generation(self,seq,node_mass):
        seq_mass = np.array([self.aa_mass_dict[aa] for aa in seq]).cumsum()
        trans_mask = torch.zeros((seq_mass.size,node_mass.size))
        trans_mask[0,0] = -float('inf')
        for i, board in enumerate(node_mass.searchsorted(seq_mass+min(self.aa_mass_dict.values())-0.02)[:-1],start=1):
            trans_mask[i,:board] = -float('inf')
        return trans_mask
    
    def trans_mask_optimum_path(self, node_mass, graph_label, dist):
        node_num = node_mass.size
        edge_mask = torch.zeros(node_num,node_num,dtype=bool)
        for x,y in enumerate(node_mass.searchsorted(node_mass+min(self.aa_mass_dict.values())*7+0.04)):
            edge_mask[x,y:] = True
        edge_mask = torch.logical_or(edge_mask,dist!=0)
        trans_mask=((graph_label@edge_mask.int())!=0).bool()
        trans_mask = torch.where(trans_mask,0.0,-float('inf'))
        return trans_mask[:-1]
    
    def seq2id(self, seq):
        return torch.LongTensor([self.aa_id['<bos>']]+[self.aa_id[aa] for aa in seq])

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