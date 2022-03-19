import torch
import torch_sparse
import torch.nn as nn

class EdgeEncoder(nn.Module):
    def __init__(self,
                 edge_type_num: int,
                 path_max_length: int,
                 d_edge: int,
                 d_ori_edge: int,
                 d_relation: int, 
                 expansion_factor: int) -> None:
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
        d_edge_type = d_edge - d_ori_edge
        d_edge_extanded = d_edge * expansion_factor
        assert d_edge % 8 == 0 #加速计算，如果是8的倍数速度可以更快.
        assert expansion_factor >= 4 #Pointnet放大乘数不因小于最后一次mlp

        self.edge_type_embed = nn.Embedding(edge_type_num, d_edge_type, padding_idx=0)
        self.edge_pos_embed = nn.Embedding(path_max_length, d_edge)
        self.dist_embed = nn.Embedding(path_max_length, d_edge_extanded, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(d_edge, d_edge*2),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_edge*2, d_edge*4),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(d_edge*4, d_edge_extanded),
                                 #nn.ReLU(inplace=True),
                                 #在这个简单的任务当中，256的句子位宽已经够用了。但是如果增加复杂度，就不一定了。
                                 #注意，此处增加位宽将会极大的增加模型内存占用。
                                 #nn.Linear(d_edge*2, d_edge_extanded),
                                 #nn.ReLU(inplace=True),
                                )
        
        self.fc = nn.Sequential(nn.ReLU(inplace=True),
                                nn.LayerNorm(d_edge_extanded),
                                nn.Linear(d_edge_extanded, d_edge*4),
                                nn.ReLU(inplace=True),
                                nn.LayerNorm(d_edge*4),
                                nn.Linear(d_edge*4, d_edge*4),
                                nn.ReLU(inplace=True),
                                nn.LayerNorm(d_edge*4),
                                nn.Linear(d_edge*4, d_relation)
                               )
        

    def forward(self, rel_type: torch.IntTensor, rel_error: torch.Tensor, edge_pos: torch.IntTensor, 
                dist: torch.IntTensor, rel_coor_cated: torch.IntTensor, batch_num: int, max_node: int):
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
        edge_pos = self.edge_pos_embed(edge_pos)
        dist = self.dist_embed(dist)
        rel_type = torch.concat([rel_type, rel_error],dim=1).add_(edge_pos)
        rel_type = self.mlp(rel_type)
        relation = torch_sparse.SparseTensor(row=rel_coor_cated[0],
                                            col=rel_coor_cated[1],
                                            value=rel_type,
                                            sparse_sizes=[batch_num*max_node*max_node,10000],
                                            is_sorted=True)
        relation = torch_sparse.reduce.max(relation,dim=1).view(batch_num, max_node, max_node, -1)
        relation = self.fc(relation+dist).permute(0,3,1,2).triu().permute(0,2,3,1)
        relation = relation + relation.transpose(1,2)
        return relation