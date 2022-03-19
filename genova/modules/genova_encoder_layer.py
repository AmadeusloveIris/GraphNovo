import torch
import torch.nn as nn

class Relation(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 d_relation: int,
                 num_head: int,
                 encoder_layer_num: int, 
                 decoder_layer_num: int):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()

        self.num_head  = num_head
        self.hidden_size = hidden_size
        assert self.hidden_size//self.num_head*self.num_head == self.hidden_size

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        self.norm_act = nn.LayerNorm(hidden_size)

        self.linear_q = nn.Linear(hidden_size, self.d_relation)
        self.linear_k = nn.Linear(hidden_size, self.d_relation)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        
        nn.init.xavier_uniform_(self.linear_q.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.linear_k.weight, gain=2**-0.5)

        self.relation_mlp = nn.Sequential(nn.LayerNorm(self.d_relation),
                                          #nn.Linear(self.d_relation, self.d_relation*2),
                                          #nn.ReLU(inplace=True),
                                          #nn.LayerNorm(self.d_relation*2),
                                          #nn.Linear(self.d_relation*2, self.d_relation),
                                          #nn.ReLU(inplace=True),
                                          #LayerNorm_out(self.d_relation),
                                          #nn.Linear(self.d_relation, num_head)
                                          nn.Linear(self.d_relation, self.d_relation//2),
                                          nn.ReLU(inplace=True),
                                          nn.LayerNorm(self.d_relation//2),
                                          nn.Linear(self.d_relation//2, self.d_relation//4),
                                          nn.ReLU(inplace=True),
                                          nn.LayerNorm(self.d_relation//4),
                                          nn.Linear(self.d_relation//4, num_head)
                                         )
        
        self.direction_embedding = nn.Embedding(3, self.d_relation) #forward(1), backward(2) and no direction(0)
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_size),
                                          nn.Linear(hidden_size, hidden_size)
                                          )
                                          
        nn.init.xavier_uniform_(self.output_layer[-1].weight, gain=encoder_layer_num**-0.25 * decoder_layer_num**-0.0625)

    def forward(self, node, edge, drctn, rel_mask):
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
        node_q = self.linear_q(node).view(batch_size, -1, self.d_relation, 1)                              # [b, len, d_srel, 1]
        node_k = self.linear_k(node).view(batch_size, -1, 1, self.d_relation)                              # [b, len, 1, d_srel]
        node_v = self.linear_v(node).view(batch_size, -1, self.num_head, self.hidden_size//self.num_head)  # [b, len, h, d_v]

        node_q = node_q.transpose(1, 2)   # [b, d_srel, len, 1]
        node_k = node_k.transpose(1, 3)   # [b, d_srel, 1, len]
        node_v = node_v.transpose(1, 2)   # [b, h, len, d_v]

        # Scaled Dot-Product Attention.
        relation = torch.matmul(node_q, node_k)                                       # [b, d_srel, q_len, k_len]
        relation = relation.permute(0,2,3,1) + edge + self.direction_embedding(drctn) # [b, q_len, k_len, d_srel]
        relation = self.relation_mlp(relation)
        
        relation += rel_mask
        relation = relation.softmax(dim=2)                          # [b, q_len, k_len, n_heads]
        relation = relation.permute(0,3,1,2)                        # [b, n_heads, q_len, k_len]
        
        node = torch.matmul(relation, node_v)
        node = node.transpose(1,2).contiguous()
        node = node.view(batch_size, -1, self.hidden_size)
        node = self.output_layer(node)
        return node

class GenovaEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, 
    ffn_hidden_size: int, d_relation: int, 
    num_head: int, encoder_layer_num: int, 
    decoder_layer_num: int):

        super().__init__()
        if decoder_layer_num<1: decoder_layer_num = 1
        if encoder_layer_num<1: encoder_layer_num = 1
        self.relation = Relation(hidden_size, d_relation, num_head, encoder_layer_num, decoder_layer_num)
        self.ffn = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ffn_hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.LayerNorm(ffn_hidden_size),
                                 nn.Linear(ffn_hidden_size, hidden_size)
                                 )
        nn.init.xavier_uniform_(self.ffn[-1].weight, gain=encoder_layer_num**-0.25 * decoder_layer_num**-0.0625)

    def forward(self, node, edge, **kwarg):
        node = node + self.relation(node, edge, **kwarg)
        node = node + self.ffn(node)
        return node