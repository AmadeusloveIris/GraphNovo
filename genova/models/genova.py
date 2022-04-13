import torch.nn as nn

from genova.models.genova_encoder import GenovaEncoder
from genova.models.genova_decoder import GenovaDecoder

class Genova(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = GenovaEncoder(cfg)
        self.decoder = GenovaDecoder(cfg)

        if self.cfg.task == 'optimum_path':
            self.query_node_linear = nn.Sequential(nn.LayerNorm(cfg.decoder.hidden_size),
                                                nn.Linear(cfg.decoder.hidden_size,cfg.decoder.hidden_size),
                                                )
            
            self.graph_node_linear = nn.Sequential(nn.LayerNorm(cfg.encoder.hidden_size),
                                                nn.Linear(cfg.encoder.hidden_size,cfg.decoder.hidden_size),
                                                )
        
        elif self.cfg.task == 'optimum_path_sequence':
            raise NotImplementedError
        
        elif self.cfg.task == 'sequence_generation':
            raise NotImplementedError
        
        elif self.cfg.task == 'node_classification':
            raise NotImplementedError
    
    def forward(self, *, encoder_input, decoder_input, graph_probability):
        graph_node = self.encoder(**encoder_input)
        if self.cfg.task == 'optimum_path':
            query_node = graph_probability@graph_node
            query_node = self.decoder(**decoder_input, tgt=query_node, graph_node=graph_node)
            query_node = self.query_node_linear(query_node)
            graph_node = self.graph_node_linear(graph_node).transpose(1,2)
            graph_probability = query_node@graph_node
            return graph_probability
