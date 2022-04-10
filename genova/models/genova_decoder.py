import torch.nn as nn

class GenovaDecoder(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.aa_embedding = nn.Embedding(cfg.decoder.dictlen,cfg.hidden_size,padding_idx=0)
        self.pos_embedding = nn.Embedding(cfg.decoder.max_seq_len,cfg.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(cfg.hidden_size,
                                                   cfg.decoder.num_heads,
                                                   batch_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.decoder.num_layers)

    def forward(self, tgt_index, memory, **kwargs):
        tgt = self.aa_embedding(tgt_index)
        tgt = tgt+self.pos_embedding.weight[:tgt_index.size(-1)]
        return self.decoder(tgt=tgt,memory=memory,**kwargs)

