import torch.nn as nn

from ..modules.edge_encoder import EdgeEncoder
from ..modules.node_encoder import NodeEncoder
from ..modules.genova_encoder_layer import GenovaEncoderLayer
class GenovaEncoder(nn.Module):
    def __init__(self, cfg, bin_classification=False):
        super().__init__()
        self.node_encoder = NodeEncoder(d_ori_node = cfg.preprocessing.d_ori_node, 
                                        max_charge = cfg.preprocessing.max_charge, 
                                        max_subion_num = cfg.preprocessing.max_subion_num, 
                                        d_node = cfg.encoder.node_encoder.d_node,
                                        expansion_factor = cfg.encoder.node_encoder.expansion_factor,
                                        hidden_size = cfg.hidden_size)

        self.edge_encoder = EdgeEncoder(edge_type_num = cfg.preprocessing.edge_type_num, 
                                        path_max_length = cfg.preprocessing.path_max_length, 
                                        d_ori_edge = cfg.preprocessing.d_ori_edge,
                                        d_edge = cfg.encoder.edge_encoder.d_edge,
                                        expansion_factor = cfg.encoder.edge_encoder.expansion_factor,
                                        d_relation = cfg.encoder.d_relation,
                                        )

        self.genova_encoder_layers = nn.ModuleList([GenovaEncoderLayer(hidden_size = cfg.hidden_size,
                                                                       ffn_hidden_size = cfg.encoder.relation.ffn_hidden_size,
                                                                       num_head = cfg.encoder.relation.num_heads,
                                                                       d_relation = cfg.encoder.d_relation,
                                                                       layer_num = cfg.encoder.num_layers)]*cfg.encoder.num_layers)
        
        self.bin_classification = bin_classification
        if self.bin_classification: 
            self.output_ffn = nn.Sequential(nn.LayerNorm(cfg.hidden_size),
                                            nn.Linear(cfg.hidden_size,cfg.hidden_size),
                                            nn.LayerNorm(cfg.hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(cfg.hidden_size,cfg.hidden_size),
                                            nn.LayerNorm(cfg.hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(cfg.hidden_size,2)
                                            )
            

    def forward(self,node_input, edge_input, rel_input):
        
        node = self.node_encoder(**node_input)
        edge = self.edge_encoder(**edge_input)
        
        for genova_encoder_layer in self.genova_encoder_layers:
            node = genova_encoder_layer(node, edge, **rel_input)
        
        if self.bin_classification:
            node = self.output_ffn(node)
        return node