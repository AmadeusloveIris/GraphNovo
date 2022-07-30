import torch.nn as nn
from genova.modules import GenovaDecoderLayer
class GenovaDecoder(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.genova_decoder_layers = nn.ModuleList([GenovaDecoderLayer(tgt_hidden_size = cfg.encoder.hidden_size,
                                                                       mem_hidden_size = cfg.decoder.hidden_size,
                                                                       d_relation = cfg.decoder.d_relation,
                                                                       decoder_layer_num = cfg.decoder.num_layers)]*cfg.decoder.num_layers)

    def forward(self, tgt, graph_node, trans_mask, self_mask):
        for genova_decoder_layer in self.genova_decoder_layers:
            tgt = genova_decoder_layer(tgt=tgt, mem=graph_node, 
                                       trans_mask=trans_mask, self_mask=self_mask)
        return tgt