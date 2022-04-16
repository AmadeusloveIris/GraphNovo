import torch
import torch.nn as nn

class Relation(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 d_relation: int,
                 gain: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        self.hidden_size = hidden_size
        assert self.hidden_size//d_relation*d_relation == self.hidden_size

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        # 使用Pre Norm，降低训练难度
        self.norm_act = nn.LayerNorm(hidden_size)

        self.linear_q = nn.Linear(hidden_size, self.d_relation)
        self.linear_k = nn.Linear(hidden_size, self.d_relation)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_edge = nn.Linear(self.d_relation, self.d_relation, bias=False)
        self.linear_path = nn.Linear(self.d_relation, self.d_relation, bias=False)

        # 因为这里我们三张图求和，所以每个位置的方差也会求和。所以在这里将每个图输出的初始值方差修正为1/3，即初始化方差乘1/sqrt(3).
        nn.init.xavier_normal_(self.linear_q.weight, gain=3**-0.25)
        nn.init.xavier_normal_(self.linear_k.weight, gain=3**-0.25)
        nn.init.xavier_normal_(self.linear_edge.weight, gain=3**-0.5)
        nn.init.xavier_normal_(self.linear_path.weight, gain=3**-0.5)
        
        # 由于使用最短或最长路径encoding图会导致每个边包含大量信息，在这里我们对Graphormer原文做出了改进
        # 我们认为graph encoding之后可以使用更长的vector表示每个node pair之间的关系
        # 并且我们使用talking head attention对node pair之间的relation ship进行编码，以克服低秩瓶颈
        self.talking = nn.Linear(self.d_relation, self.d_relation, bias=False)

        self.output_layer = nn.Linear(hidden_size, hidden_size)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_v.weight, gain=gain)
        nn.init.xavier_normal_(self.output_layer.weight, gain=gain)

    def forward(self, node, edge, path, rel_mask):
        """_summary_

        Args:
            node (Tensor): node information from last layer
            edge (Tensor): edge information from edge encoder
            drctn (IntTensor): direction mark
            rel_mask (Tensor): relation mask for ignore some pair of nodes which don't have any connection

        Returns:
            node (Tensor): node information from last layer
        """
        batch_size = node.size(0)
        
        node = self.norm_act(node)
        node_q = self.linear_q(node).view(batch_size, -1, self.d_relation, 1)                                # [b, len, d_srel, 1]
        node_k = self.linear_k(node).view(batch_size, -1, 1, self.d_relation)                                # [b, len, 1, d_srel]
        node_v = self.linear_v(node).view(batch_size, -1, self.d_relation, self.hidden_size//self.d_relation)  # [b, len, h, d_v]

        node_q = node_q.transpose(1, 2)   # [b, d_srel, len, 1]
        node_k = node_k.transpose(1, 3)   # [b, d_srel, 1, len]
        node_v = node_v.transpose(1, 2)   # [b, h, len, d_v]

        # Scaled Dot-Product Attention.
        relation = torch.matmul(node_q, node_k)                                                 # [b, d_srel, q_len, k_len]
        relation = relation.permute(0,2,3,1) + self.linear_edge(edge) + self.linear_path(path)  # [b, q_len, k_len, d_srel]
        relation = self.talking(relation)
        relation += rel_mask
        relation = relation.softmax(dim=2)                          # [b, q_len, k_len, d_srel]
        relation = relation.permute(0,3,1,2)                        # [b, d_srel, q_len, k_len]
        
        node = torch.matmul(relation, node_v)
        node = node.transpose(1,2).reshape(batch_size, -1, self.hidden_size)
        node = self.output_layer(node)
        return node

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int, gain: float):
        super().__init__()

        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        self.ln = nn.LayerNorm(hidden_size)
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 4*hidden_size, bias=False),
                                          nn.GELU()
                                          )
        self.pre_ffn = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.ffnln = nn.LayerNorm(4*hidden_size)
        self.post_ffn = nn.Linear(4*hidden_size, hidden_size, bias=False)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.post_ffn.weight, gain=gain)

    def forward(self, x):
        x = self.ln(x)
        x = self.ffnln(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.post_ffn(x)
        return x

class GenovaEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int,
    d_relation: int,
    encoder_layer_num: int, 
    decoder_layer_num: int = 1):

        super().__init__()
        gain = encoder_layer_num**-0.5 * decoder_layer_num**-0.25
        self.relation = Relation(hidden_size, d_relation, gain)
        self.ffn = FFNGLU(hidden_size)

    def forward(self, node, edge, path, rel_mask):
        node = node + self.relation(node, edge, path, rel_mask)
        node = node + self.ffn(node)
        return node