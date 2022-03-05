import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    def __init__(self,
                 d_ori_node: int,
                 max_charge: int,
                 max_subion_num: int,
                 d_node: int,
                 expansion_factor: int, 
                 hidden_size: int) -> None:
        super().__init__()
        d_ion_embed = d_node - d_ori_node
        d_node_extanded = d_node * expansion_factor
        d_node_compress = (d_node_extanded + hidden_size)//2
        assert d_node % 8 == 0
        assert expansion_factor >= 8
        assert d_node_compress <= d_node_extanded and d_node_compress >= hidden_size and d_node_compress%8 == 0

        self.ion_source_embed = nn.Embedding(max_subion_num, d_ion_embed,padding_idx=0)
        self.charge_embed = nn.Embedding(max_charge, d_node_extanded)
        self.shared_mlp = nn.Sequential(nn.Linear(d_node, d_node*2),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(d_node*2),
                                        nn.Linear(d_node*2, d_node*4),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(d_node*4),
                                        nn.Linear(d_node*4, d_node*8),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(d_node*8),
                                        nn.Linear(d_node*8, d_node_extanded),
                                       )
        
        self.fc = nn.Sequential(nn.ReLU(inplace=True),
                                nn.LayerNorm(d_node_extanded),
                                nn.Linear(d_node_extanded, d_node_compress),
                                nn.ReLU(inplace=True),
                                nn.LayerNorm(d_node_compress),
                                nn.Linear(d_node_compress, d_node_compress),
                                nn.ReLU(inplace=True),
                                nn.LayerNorm(d_node_compress),
                                nn.Linear(d_node_compress, hidden_size)
                               )
    
    def forward(self, node_feat, node_sourceion, charge):
        node_feat_source = self.ion_source_embed(node_sourceion)
        charge_embedding = self.charge_embed(charge).unsqueeze(1)
        node_feat = torch.concat([node_feat, node_feat_source], dim=-1)
        node = self.shared_mlp(node_feat)
        node, _ = torch.max(node, -2)
        node = self.fc(node + charge_embedding)
        return node