import torch
import torch.nn as nn
import torch.nn.functional as F

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = torch.einsum('...r,hr -> ...hr', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,
        max_peaks = 256,
        query_key_dim = 128,
        expansion_factor = 2.,
        dropout = 0.
    ):
        super().__init__()
        self.max_peaks = max_peaks
        hidden_dim = int(expansion_factor * dim)
        self.hidden_dim = hidden_dim
        self.query_key_dim = query_key_dim
        self.to_hidden = nn.Sequential(nn.LayerNorm(dim),
                                       nn.Linear(dim, hidden_dim * 2 + query_key_dim),
                                       nn.SiLU()
                                      )
        
        self.offsetscale = OffsetScale(query_key_dim, heads = 2)

        if dropout>0:
            self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim),
                                        nn.Dropout(dropout)
                                       )
        else:
            self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self,x,*,peaks_moverz,src_key_padding_mask):
        uv_base = self.to_hidden(x)
        u,v,base = uv_base.split([self.hidden_dim, self.hidden_dim, self.query_key_dim],dim=-1)
        q, k = self.offsetscale(base)
        q = self.apply_moverz_distance(q, peaks_moverz)
        k = self.apply_moverz_distance(k, peaks_moverz)
        qk = torch.einsum('bnr,bmr->bnm', q, k)
        qk = qk.masked_fill(src_key_padding_mask,0)
        attn = F.relu(qk)**2/(self.max_peaks*self.query_key_dim)
        out = torch.einsum('bnm,bmd->bnd', attn, v) * u
        out = self.to_out(out) + x
        return out
    
    @staticmethod
    def apply_moverz_distance(x, peaks_moverz):
        peaks_moverz_sin, peaks_moverz_cos = peaks_moverz
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.concat([x0*peaks_moverz_cos-x1*peaks_moverz_sin,\
                             x1*peaks_moverz_cos+x0*peaks_moverz_sin], dim = -1)