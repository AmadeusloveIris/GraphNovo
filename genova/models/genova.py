import torch.nn as nn
from genova.models import GenovaEncoder, GenovaDecoder

class Genova(nn.Module):
    def __init__(self,cfg,dict_len=23) -> None:
        super().__init__()
        self.cfg = cfg
        self.dict_len = dict_len
        self.encoder = GenovaEncoder(cfg)

        if self.cfg.task == 'optimum_path':
            self.decoder = GenovaDecoder(cfg)
            self.query_node_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                                nn.Linear(cfg.decoder.hidden_size,cfg.decoder.hidden_size),
                                                )
            
            self.graph_node_linear = nn.Sequential(nn.LayerNorm(cfg.encoder.hidden_size),
                                                nn.Linear(cfg.encoder.hidden_size,cfg.decoder.hidden_size),
                                                )
        
        elif self.cfg.task == 'optimum_path_sequence':
            raise NotImplementedError
        
        elif self.cfg.task == 'sequence_generation':
            self.decoder = GenovaDecoder(cfg)
            self.tgt_embedding = nn.Embedding(dict_len,self.cfg.decoder.hidden_size,padding_idx=0)
            self.pos_embedding = nn.Embedding(32,self.cfg.decoder.hidden_size)
            self.output_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                               nn.Linear(cfg.decoder.hidden_size,dict_len)
                                               )
        
        elif self.cfg.task == 'node_classification':
            self.output_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                               nn.Linear(cfg.decoder.hidden_size,1)
                                               )
    
    def forward(self, *, encoder_input, decoder_input=None, tgt=None):
        graph_node = self.encoder(**encoder_input)
        if self.cfg.task == 'optimum_path':
            assert decoder_input and tgt
            query_node = tgt@graph_node
            query_node = self.decoder(**decoder_input, tgt=query_node, graph_node=graph_node)
            query_node = self.query_node_linear(query_node)
            graph_node = self.graph_node_linear(graph_node).transpose(1,2)
            graph_probability = query_node@graph_node
            graph_probability = graph_probability+decoder_input['trans_mask'].squeeze(-1)
            return graph_probability
        elif self.cfg.task == 'sequence_generation':
            assert decoder_input and tgt
            tgt = self.tgt_embedding(tgt['seq']) + self.pos_embedding(tgt['pos'])
            tgt = self.decoder(**decoder_input, tgt=tgt, graph_node=graph_node)
            tgt = self.output_linear(tgt)
            return tgt
        elif self.cfg.task == 'node_classification':
            tgt = self.output_linear(graph_node)
            return tgt

