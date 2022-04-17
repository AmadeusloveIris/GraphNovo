import torch
import torch.nn as nn

class MaskedSelfRelation(nn.Module):
    def __init__(self,
                 tgt_hidden_size: int,
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
        self.tgt_hidden_size = tgt_hidden_size
        assert self.tgt_hidden_size//d_relation*d_relation == self.tgt_hidden_size

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        # 使用Pre Norm，降低训练难度
        self.norm_act = nn.LayerNorm(tgt_hidden_size)

        self.linear_q = nn.Linear(tgt_hidden_size, self.d_relation)
        self.linear_k = nn.Linear(tgt_hidden_size, self.d_relation)
        self.linear_v = nn.Linear(tgt_hidden_size, tgt_hidden_size)
        
        # 使用talking head attention对node pair之间的relation ship进行编码，以克服低秩瓶颈
        self.talking = nn.Linear(self.d_relation, self.d_relation, bias=False)

        self.output_layer = nn.Linear(tgt_hidden_size, tgt_hidden_size)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_v.weight, gain=gain)
        nn.init.xavier_normal_(self.output_layer.weight, gain=gain)

    def forward(self, tgt, rel_mask):
        """_summary_

        Args:
            node (Tensor): node information from last layer
            edge (Tensor): edge information from edge encoder
            drctn (IntTensor): direction mark
            rel_mask (Tensor): relation mask for ignore some pair of nodes which don't have any connection

        Returns:
            node (Tensor): node information from last layer
        """
        batch_size = tgt.size(0)
        
        tgt = self.norm_act(tgt)
        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.d_relation, 1)                                # [b, len, d_srel, 1]
        tgt_k = self.linear_k(tgt).view(batch_size, -1, 1, self.d_relation)                                # [b, len, 1, d_srel]
        tgt_v = self.linear_v(tgt).view(batch_size, -1, self.d_relation, self.tgt_hidden_size//self.d_relation)  # [b, len, h, d_v]

        tgt_q = tgt_q.transpose(1, 2)   # [b, d_srel, q_len, 1]
        tgt_k = tgt_k.transpose(1, 3)   # [b, d_srel, 1, k_len]
        tgt_v = tgt_v.transpose(1, 2)   # [b, h, len, d_v]

        # Scaled Dot-Product Attention.
        relation = torch.matmul(tgt_q, tgt_k) # [b, d_srel, q_len, k_len]
        relation = relation.permute(0,2,3,1)  # [b, q_len, k_len, d_srel]
        relation = self.talking(relation)     # [b, q_len, k_len, d_srel]
        relation += rel_mask
        relation = relation.softmax(dim=2)                          # [b, q_len, k_len, n_heads]
        relation = relation.permute(0,3,1,2)                        # [b, n_heads, q_len, k_len]
        
        tgt_v = torch.matmul(relation, tgt_v)
        tgt_v = tgt_v.transpose(1,2).reshape(batch_size, -1, self.tgt_hidden_size)
        tgt_v = self.output_layer(tgt_v)
        return tgt_v

class TransRelation(nn.Module):
    def __init__(self,
                 tgt_hidden_size: int,
                 mem_hidden_size: int,
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
        self.tgt_hidden_size = tgt_hidden_size
        assert self.tgt_hidden_size//d_relation*d_relation == self.tgt_hidden_size

        self.d_relation = d_relation
        assert self.d_relation % 8 == 0

        # 使用Pre Norm，降低训练难度
        self.q_norm_act = nn.LayerNorm(tgt_hidden_size)
        self.k_norm_act = nn.LayerNorm(mem_hidden_size)

        self.linear_q = nn.Linear(tgt_hidden_size, self.d_relation)
        self.linear_k = nn.Linear(mem_hidden_size, self.d_relation)
        self.linear_v = nn.Linear(mem_hidden_size, tgt_hidden_size)
        
        # 使用talking head attention对node pair之间的relation ship进行编码，以克服低秩瓶颈
        self.talking = nn.Linear(self.d_relation, self.d_relation, bias=False)

        self.output_layer = nn.Linear(tgt_hidden_size, tgt_hidden_size)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.xavier_normal_(self.linear_k.weight)
        nn.init.xavier_normal_(self.linear_v.weight, gain=gain)
        nn.init.xavier_normal_(self.output_layer.weight, gain=gain)

    def forward(self, tgt, mem, rel_mask):
        """_summary_

        Args:
            node (Tensor): node information from last layer
            edge (Tensor): edge information from edge encoder
            drctn (IntTensor): direction mark
            rel_mask (Tensor): relation mask for ignore some pair of nodes which don't have any connection

        Returns:
            node (Tensor): node information from last layer
        """
        batch_size = tgt.size(0)
        
        tgt = self.q_norm_act(tgt)
        mem = self.k_norm_act(mem)
        
        tgt_q = self.linear_q(tgt).view(batch_size, -1, self.d_relation, 1)                                # [b, len, d_srel, 1]
        mem_k = self.linear_k(mem).view(batch_size, -1, 1, self.d_relation)                                # [b, len, 1, d_srel]
        mem_v = self.linear_v(mem).view(batch_size, -1, self.d_relation, self.tgt_hidden_size//self.d_relation)  # [b, len, h, d_v]

        tgt_q = tgt_q.transpose(1, 2)   # [b, d_srel, q_len, 1]
        mem_k = mem_k.transpose(1, 3)   # [b, d_srel, 1, k_len]
        mem_v = mem_v.transpose(1, 2)   # [b, h, len, d_v]

        # Scaled Dot-Product Attention.
        relation = torch.matmul(tgt_q, mem_k)   # [b, d_srel, q_len, k_len]
        relation = relation.permute(0,2,3,1)    # [b, q_len, k_len, d_srel]
        relation = self.talking(relation)
        relation += rel_mask
        relation = relation.softmax(dim=2)                          # [b, q_len, k_len, d_srel]
        relation = relation.permute(0,3,1,2)                        # [b, d_srel, q_len, k_len]
        
        mem_v = torch.matmul(relation, mem_v)
        mem_v = mem_v.transpose(1,2).reshape(batch_size, -1, self.tgt_hidden_size)
        mem_v = self.output_layer(mem_v)
        return mem_v

class FFNGLU(nn.Module):
    def __init__(self, tgt_hidden_size: int, gain: float):
        super().__init__()

        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        self.ln = nn.LayerNorm(tgt_hidden_size)
        self.pre_ffn_gate = nn.Sequential(nn.Linear(tgt_hidden_size, 4*tgt_hidden_size, bias=False),
                                          nn.GELU()
                                          )
        self.pre_ffn = nn.Linear(tgt_hidden_size, 4*tgt_hidden_size, bias=False)
        self.ffnln = nn.LayerNorm(4*tgt_hidden_size)
        self.post_ffn = nn.Linear(4*tgt_hidden_size, tgt_hidden_size, bias=False)
        
        #根据DeepNet，对初始化值做修正.
        nn.init.xavier_normal_(self.post_ffn.weight, gain=gain)

    def forward(self, x):
        x = self.ln(x)
        x = self.ffnln(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.post_ffn(x)
        return x

class GenovaDecoderLayer(nn.Module):
    def __init__(
        self, tgt_hidden_size: int,
        mem_hidden_size: int,
        d_relation: int, 
        decoder_layer_num: int = 1):

        super().__init__()
        gain = decoder_layer_num**-0.25
        self.self_relation = MaskedSelfRelation(tgt_hidden_size, d_relation, gain)
        self.trans_relation = TransRelation(tgt_hidden_size, mem_hidden_size, d_relation, gain)
        self.ffn = FFNGLU(tgt_hidden_size, gain)

    def forward(self, *, tgt, mem, trans_mask, self_mask):
        tgt = tgt + self.self_relation(tgt, self_mask)
        tgt = tgt + self.trans_relation(tgt, mem, trans_mask)
        tgt = tgt + self.ffn(tgt)
        return tgt