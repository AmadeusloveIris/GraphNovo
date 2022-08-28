import torch
import torch.nn as nn
from math import sqrt

class MaskedMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_head: int,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        assert hidden_size % 8 == 0
        assert hidden_size % n_head == 0

        self.initial = False

        self.n_head = n_head
        self.head_size = hidden_size//n_head

        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 3*hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tgt, tgt_mask, past_tgt=None):
        if self.training:
            batch_size = tgt.size(0)
            q, k, v = self.linear(tgt).view(batch_size, -1, self.n_head, 3*self.head_size).chunk(3,dim=-1)
            attn = torch.einsum('bnij,bmij->binm',q,k)/sqrt(self.head_size)
            attn = attn.masked_fill(tgt_mask, -float('inf')).softmax(dim=-1)
            x = torch.einsum('binm,bmij->bnij',attn,v).flatten(2,3)
            tgt = self.norm(self.dropout(self.output_layer(x)) + tgt)
            return tgt
        else:
            if self.initial:
                batch_size = tgt.size(0)
                q, k, v = self.linear(tgt).view(batch_size, -1, self.n_head, 3*self.head_size).chunk(3,dim=-1)
                past_tgt = k, v
                attn = torch.einsum('bnij,bmij->binm',q,k)/sqrt(self.head_size)
                attn = attn.masked_fill(tgt_mask, -float('inf')).softmax(dim=-1)
                x = torch.einsum('binm,bmij->bnij',attn,v).flatten(2,3)
                tgt = self.norm(self.dropout(self.output_layer(x)) + tgt)
                return tgt, past_tgt
            else:
                batch_size = tgt.size(0)
                k_past, v_past = past_tgt
                q, k, v = self.linear(tgt).view(batch_size, -1, self.n_head, 3*self.head_size).chunk(3,dim=-1)
                k = torch.concat([k_past,k], dim=1)
                v = torch.concat([v_past,v], dim=1)
                past_tgt = k, v
                attn = torch.einsum('bnij,bmij->binm',q,k)/sqrt(self.head_size)
                attn = attn.masked_fill(tgt_mask, -float('inf')).softmax(dim=-1)
                x = torch.einsum('binm,bmij->bnij',attn,v).flatten(2,3)
                tgt = self.norm(self.dropout(self.output_layer(x)) + tgt)
                return tgt, past_tgt

    def inference_initial(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("inference_initial mode is expected to be boolean")
        self.initial = mode

class MultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_head: int,
                 dropout_rate: float):
        """_summary_

        Args:
            hidden_size (int): same with transformer hidden_size
            d_relation (int): relation matrix dimention
            num_head (int): same with transformer num_head
            layer_num (int): How many layers in total
        """
        super().__init__()
        assert hidden_size % 8 == 0
        assert hidden_size % n_head == 0
        assert n_head % 4 == 0

        self.initial = False

        self.n_head = n_head
        self.head_size = hidden_size//n_head

        # 使用Pre Norm，降低训练难度
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_kv = nn.Linear(hidden_size, 2*hidden_size)

        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tgt, mem_key_padding_mask, pep_mass, mem=None, peaks_moverz=None, past_mem=None):
        """_summary_

        Args:
            tgt (_type_): _description_
            mem_key_padding_mask (_type_): _description_
            pep_mass (_type_): _description_
            mem (_type_, optional): _description_. Defaults to None.
            peaks_moverz (_type_, optional): _description_. Defaults to None.
            past_mem (_type_, optional): _description_. Defaults to None.

        Raises:
            TypeError: _description_
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        batch_size = tgt.size(0)
        
        q = self.linear_q(tgt).view(batch_size, -1, self.n_head, self.head_size)
        q = self.apply_moverz_distance(q, pep_mass)
        
        if self.training:
            k, v = self.linear_kv(mem).view(batch_size, -1, self.n_head, self.head_size*2).chunk(2,dim=-1)
            k = self.apply_moverz_distance(k, peaks_moverz)
        else:
            if self.initial:
                k, v = self.linear_kv(mem).view(batch_size, -1, self.n_head, self.head_size*2).chunk(2,dim=-1)
                k = self.apply_moverz_distance(k, peaks_moverz)
                past_mem = k,v
            else:
                k,v = past_mem

        attn = torch.einsum('bnij,bmij->binm',q,k)/sqrt(self.head_size)
        attn = attn.masked_fill(mem_key_padding_mask,-float('inf')).softmax(dim=-1)
        x = torch.einsum('binm,bmij->bnij',attn,v).flatten(2,3)
        tgt = self.norm(self.dropout(self.output_layer(x)) + tgt)
        if self.training: return tgt
        else: 
            if self.initial: return tgt, past_mem
            else: return tgt

    def inference_initial(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("inference_initial mode is expected to be boolean")
        self.initial = mode
    
    @staticmethod
    def apply_moverz_distance(x, peaks_moverz):
        peaks_moverz_sin, peaks_moverz_cos = peaks_moverz
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*peaks_moverz_cos-x1*peaks_moverz_sin,\
                             x1*peaks_moverz_cos+x0*peaks_moverz_sin], dim = -1)

class FFN(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(hidden_size,4*hidden_size,bias=False),
                                   nn.ReLU(),
                                   nn.Linear(4*hidden_size,hidden_size,bias=False)
                                  )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        return self.norm(x+self.dropout(self.layer(x)))

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, n_head, dropout_rate):
        super().__init__()
        self.initial = False
        self.mmha = MaskedMultiHeadAttn(hidden_size,n_head,dropout_rate)
        self.mha = MultiHeadAttn(hidden_size,n_head,dropout_rate)
        self.ffn = FFN(hidden_size,dropout_rate)

    def forward(self, tgt, *, mem_key_padding_mask, pep_mass, tgt_mask, mem=None, peaks_moverz=None, past_mem=None, past_tgt=None):
        if self.training:
            tgt = self.mmha(tgt, tgt_mask)
            tgt = self.mha(tgt, mem_key_padding_mask=mem_key_padding_mask, pep_mass=pep_mass, mem=mem, peaks_moverz=peaks_moverz)
            tgt = self.ffn(tgt)
            return tgt
        else:
            if self.initial:
                tgt, past_tgt = self.mmha(tgt, tgt_mask)
                tgt, past_mem = self.mha(tgt, mem_key_padding_mask=mem_key_padding_mask, pep_mass=pep_mass, mem=mem, peaks_moverz=peaks_moverz)
                tgt = self.ffn(tgt)
                return tgt, past_tgt, past_mem
            else:
                tgt, past_tgt = self.mmha(tgt, tgt_mask, past_tgt)
                tgt = self.mha(tgt, mem_key_padding_mask=mem_key_padding_mask, pep_mass=pep_mass, past_mem=past_mem)
                tgt = self.ffn(tgt)
                return tgt, past_tgt

    def inference_initial(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("inference_initial mode is expected to be boolean")
        self.initial = mode
        for module in self.children():
            if hasattr(module, "inference_initial"): module.inference_initial(mode)
