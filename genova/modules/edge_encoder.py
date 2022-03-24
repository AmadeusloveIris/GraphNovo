import torch
import torch_sparse
import torch.nn as nn

class EdgeEncoder(nn.Module):
    def __init__(self,
                 edge_type_num: int,
                 d_edge: int,
                 d_ori_edge: int,
                 d_relation: int, 
                 expansion_factor: int,
                 path_max_length: int = None) -> None:
        """EdgeEncoder负责将graph中的edge进行编码，包括edge_type, edge_difference, edge在longest path中的位置信息。
        并且最终会将整个句子的长度加入进去。
        
        Args:
            d_embedding_edge (int): Edge type embedding dimension size
            d_relation (int):  最终进入Relationship matrix dimention
            expansion_factor (int): Pointnet放大乘数
        
        Key parameter:
            d_edge_ori: 由于embedding后还要和difference做concat，所以要加一维。
            d_edge_extanded: pointnet需要足够多的维度做maxpool，以避免信息丢失。这是设置放大维度的。
        """
        super().__init__()
        self.edge_type_num = edge_type_num
        d_edge_type = d_edge - d_ori_edge
        self.d_edge_extanded = d_edge * expansion_factor
        self.path_max_length = path_max_length
        assert d_edge % 8 == 0 #加速计算，如果是8的倍数速度可以更快.
        assert expansion_factor >= 4 #Pointnet放大乘数不因小于最后一次mlp

        self.edge_type_embed = nn.Embedding(edge_type_num, d_edge_type, padding_idx=0)
        if path_max_length:
            self.edge_pos_embed = nn.Embedding(path_max_length, d_edge)
            self.dist_embed = nn.Embedding(path_max_length, self.d_edge_extanded, padding_idx=0)
            self.mlp = nn.Sequential(nn.Linear(d_edge, d_edge*2),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(d_edge*2, d_edge*4),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(d_edge*4, self.d_edge_extanded)
                                    )
        else:
            self.mlp = nn.Sequential(nn.Linear(d_edge, d_edge*2),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(d_edge*2, self.d_edge_extanded)
                                    )
        
        self.fc = nn.Sequential(nn.LayerNorm(self.d_edge_extanded),
                                nn.Linear(self.d_edge_extanded, d_edge*4),
                                nn.ReLU(inplace=True),
                                nn.LayerNorm(d_edge*4),
                                nn.Linear(d_edge*4, d_relation),
                                nn.LayerNorm(d_relation)
                               )
        

    def forward(self, rel_type: torch.IntTensor, rel_error: torch.Tensor, 
                rel_coor_cated: torch.IntTensor, batch_num: int, max_node: int, 
                rel_pos: torch.IntTensor = None, dist: torch.IntTensor = None):
        """[summary]

        Args:
            rel_type (torch.IntTensor): edge type
            rel_error (torch.Tensor): edge difference
            edge_pos (torch.IntTensor): edge position in LP(Longest Path)
            dist (torch.IntTensor): length of LP
            rel_coor_cated (torch.IntTensor): coordinates of those data (Because we use Hybrid COO to store sparse tensor)
            batch_num (int): batch_num
            max_node (int): max_node in batch of graph

        Returns:
            relation (Tensor): Bias of Attention Matrix in Transformer
        """
        rel_type = self.edge_type_embed(rel_type)
        relation = torch.concat([rel_type, rel_error],dim=1)
        if rel_pos!=None: relation = relation + self.edge_pos_embed(rel_pos)
        relation = self.mlp(relation)
        if self.path_max_length:
            relation = torch_sparse.SparseTensor(row=rel_coor_cated[0],
                                                col=rel_coor_cated[1],
                                                value=relation,
                                                sparse_sizes=[batch_num*max_node*max_node,self.edge_type_num*self.d_edge_extanded*self.path_max_length],
                                                is_sorted=True)
        else:
            relation = torch_sparse.SparseTensor(row=rel_coor_cated[0],
                                                col=rel_coor_cated[1],
                                                value=relation,
                                                sparse_sizes=[batch_num*max_node*max_node,self.edge_type_num*self.d_edge_extanded],
                                                is_sorted=True)
                                                
        relation = torch_sparse.reduce.max(relation,dim=1).view(batch_num, max_node, max_node, -1)
        if dist!=None: relation = relation + self.dist_embed(dist)
        relation = self.fc(relation)
        return relation