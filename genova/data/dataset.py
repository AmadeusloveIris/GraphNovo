import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.loc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))

        spec['node_input']['charge'] = spec_head['Charge']
        seq = spec_head['Annotated Sequence']
        #spec.pop('node_mass')
        if self.cfg.task == 'node_classification':
            spec['graph_label'] = torch.any(spec['graph_label'], -1).long()
            return spec
        
        elif self.cfg.task == 'optimum path sequence':
            raise NotImplementedError
            
        elif self.cfg.task == 'sequence generation':
            raise NotImplementedError
        
        elif self.cfg.task == 'optimum path':
            graph_label = spec.pop('graph_label').T
            graph_propobility = graph_label/torch.where(graph_label.sum(-1)==0,1,graph_label.sum(-1)).unsqueeze(1)
            graph_propobility = graph_propobility[torch.any(graph_label,-1)]
            return spec, graph_label
            
            
        #spec['graph_label'] = torch.any(spec['graph_label'], -1).long()
        return spec, target

    def trans_mask_sequence_generation(self, seq, spec):
        seq_mass = genova.utils.BasicClass.Residual_seq(seq[:-1].replace('L','I')).step_mass-0.02
        memory_mask = np.zeros((seq_mass.size+1,spec['node_mass'].size))
        for i, board in enumerate(spec['node_mass'].searchsorted(seq_mass),start=1):
            memory_mask[i,:board] = -float('inf')
        memory_mask = np.repeat(memory_mask[np.newaxis],self.cfg.decoder.num_heads,axis=0)
        return memory_mask
    
    def seq2seqblock(self, ):
        seq_block = []
        for i, combine_flag in enumerate(~spec['graph_label'].any(0)[1:]):
            if combine_flag:
                if 'combine_start_index' not in locals():
                    combine_start_index = i
            else:
                try:
                    seq_block.append(seq[combine_start_index:i+1])
                    del(combine_start_index)
                except:
                    seq_block.append(seq[i])
        return seq_block
    
    def __len__(self):
        return len(self.spec_header)