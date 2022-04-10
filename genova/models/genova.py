import torch.nn as nn

from genova.models.genova_encoder import GenovaEncoder
from genova.models.genova_decoder import GenovaDecoder

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
        node = self.encoder(**encoder_input)
        pepseq = self.decoder(**decoder_input,memory=node)
        pepseq = self.output_ffn(pepseq)
        return pepseq

