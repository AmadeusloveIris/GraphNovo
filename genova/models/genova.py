import logging

import torch
import torch.nn as nn

from .genova_encoder import GenovaEncoder
from .genova_decoder import GenovaDecoder

class Genova(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.encoder = GenovaEncoder(cfg)
        self.decoder = GenovaDecoder(cfg)
        self.output_ffn = nn.Sequential(nn.LayerNorm(cfg.hidden_size),
                                        nn.Linear(cfg.hidden_size,cfg.hidden_size),
                                        nn.LayerNorm(cfg.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(cfg.hidden_size,cfg.hidden_size),
                                        nn.LayerNorm(cfg.hidden_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(cfg.hidden_size,cfg.decoder.dictlen))
    
    def forward(self, *, decoder_input, encoder_input):
        """Genova forward function.
        注意：*在这里的含义是在此之后的所有变量必须使用keyword方式输入，*本身
        不是变量。
        """
        node = self.encoder(**encoder_input)
        pepseq = self.decoder(**decoder_input,memory=node)
        pepseq = self.output_ffn(pepseq)
        return pepseq

