import torch
import torch.nn as nn


class NodeEncoder(nn.Module):
    def __init__(self,
                 d_ori_node: int,
                 max_charge: int,
                 max_rt_bin: int,
                 max_subion_num: int,
                 d_node: int,
                 expansion_factor: int, 
                 hidden_size: int) -> None:
        """A T-net structure(Pointnet) for summarizing node information

        Args:
            d_ori_node (int): 指示node来源的ion种类
            max_charge (int): 数据集中最大的charge个数（占位，不需要修改）
            max_subion_num (int): 数据集中最大的subion个数（占位，不需要修改）
            d_node (int): 每个node拼接上来源ion的embedding后，准备进入ffn时的dimension。
            expansion_factor (int): Pointnet中的最大维度乘数
            hidden_size (int): 同transformer
        """
        super().__init__()
        d_ion_embed = d_node - d_ori_node
        d_node_extanded = d_node * expansion_factor
        d_node_compress = (d_node_extanded + hidden_size)//2
        assert d_node % 8 == 0
        assert expansion_factor >= 8
        assert d_node_compress <= d_node_extanded and d_node_compress >= hidden_size and d_node_compress%8 == 0

        self.ion_source_embed = nn.Embedding(max_subion_num, d_ion_embed,padding_idx=0)
        self.charge_embed = nn.Embedding(max_charge, d_node_extanded)
        self.rt_embed = nn.Embedding(max_rt_bin+1, d_node_extanded)
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_node, d_node*2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node*2),
            nn.Linear(d_node*2, d_node*4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node*4),
            nn.Linear(d_node*4, d_node*8),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node*8),
            nn.Linear(d_node*8, d_node_extanded)
            )
        
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node_extanded),
            nn.Linear(d_node_extanded, d_node_compress),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node_compress),
            nn.Linear(d_node_compress, d_node_compress),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_node_compress),
            nn.Linear(d_node_compress, hidden_size)
            )
    
    def forward(self, node_feat, node_sourceion, charge, rt):
        """_summary_

        Args:
            node_feat (Tensor): node信息，除了来源ion
            node_sourceion (IntTensor): node来源ion
            charge (IntTensor): 原始谱图母离子电荷数

        Returns:
            node (Tensor): 同transormer word in sentence
        """
        node_feat_source = self.ion_source_embed(node_sourceion)
        charge_embedding = self.charge_embed(charge).unsqueeze(1)
        rt_embedding = self.rt_embed(rt).unsqueeze(1)
        node_feat = torch.concat([node_feat, node_feat_source], dim=-1)
        node = self.shared_mlp(node_feat)
        node, _ = torch.max(node, -2)
        node = self.fc(node + charge_embedding + rt_embedding)
        return node