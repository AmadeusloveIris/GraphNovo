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
        with open(os.path.join(self.dataset_dir_path, spec_head['Serialized File Name']), 'rb') as f:
            f.seek(spec_head['Serialized File Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['Serialized Data Length'])))

        spec['node_input']['charge'] = spec_head['Charge']
        spec.pop('node_mass')
        spec['graph_label'] = torch.any(spec['graph_label'], -1).long()
        return spec

    def __len__(self):
        return len(self.spec_header)