import os
import gzip
import torch
import pickle
from torch.utils.data import Dataset

class GenovaDataset(Dataset):
    def __init__(self, cfg, *, dictionary, spec_header, dataset_dir_path):
        super().__init__()
        self.cfg = cfg
        self.dictionary = dictionary
        self.spec_header = spec_header
        self.dataset_dir_path = dataset_dir_path
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = dict(self.spec_header.iloc[idx])
        with open(os.path.join(self.dataset_dir_path, spec_head['Serialized File Name']),'rb') as f:
            f.seek(spec_head['Serialized File Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['Serialized Data Length'])))
        
        spec['charge'] = spec_head['Charge']
        if self.cfg.dataset.use_path_label:
            decoder_input = {}
            #edge sparse data
            edge_type = spec.pop('edge_type')
            edge_error = spec.pop('edge_error')
            edge_coor = torch.stack(torch.where(edge_type>0))
            edge_error = edge_error[edge_type>0]
            edge_type = edge_type[edge_type>0]
            decoder_input['edge_type'] = edge_type
            decoder_input['edge_error'] = edge_error
            decoder_input['edge_coor'] = edge_coor
            #edge mask
            edge_mask = torch.where(spec.pop('edge_mask').bool(),0.0,1.0)
            path_label = spec.pop('path_label').T.contiguous()
            path_label = path_label[path_label.any(-1)]
            decoder_input['edge_attn_mask'] = torch.where((path_label[:-1]@edge_mask).bool(),0.0,-float('inf'))
            edge_mask = edge_mask-torch.eye(edge_mask.shape[0])
            decoder_input['edge_label_mask'] = torch.where((path_label[:-1]@edge_mask).bool(),0.0,-float('inf'))
            path_label = path_label/path_label.sum(-1).unsqueeze(-1)
            decoder_input['path_label'] = path_label[:-1]
            
            return spec, decoder_input, path_label[1:]
        else:
            spec.pop('edge_type')
            spec.pop('edge_error')
            spec.pop('edge_mask')
            spec.pop('path_label')
            seq = self.tokenizer(spec_head['Annotated Sequence'].replace('L','I'))
            return spec, seq
    
    def __len__(self):
        return len(self.spec_header)
    
    def tokenizer(self, seq):
        seq = ['<n_term>']+[aa for aa in seq]+['<c_term>']
        return torch.LongTensor([self.dictionary[aa] for aa in seq])