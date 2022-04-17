import torch.nn as nn

from genova.modules.edge_encoder import EdgeEncoder
from genova.modules.node_encoder import NodeEncoder
from genova.modules.genova_encoder_layer import GenovaEncoderLayer
class GenovaEncoder(nn.Module):
    def __init__(self, cfg):
        """_summary_

        Args:
            cfg (_type_): _description_
            bin_classification (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.node_encoder = NodeEncoder(d_ori_node = cfg.preprocessing.d_ori_node, 
                                        max_charge = cfg.preprocessing.max_charge, 
                                        max_subion_num = cfg.preprocessing.max_subion_num, 
                                        d_node = cfg.encoder.node_encoder.d_node,
                                        expansion_factor = cfg.encoder.node_encoder.expansion_factor,
                                        hidden_size = cfg.encoder.hidden_size)

        self.path_encoder = EdgeEncoder(edge_type_num = cfg.preprocessing.edge_type_num, 
                                        path_max_length = cfg.preprocessing.path_max_length, 
                                        d_ori_edge = cfg.preprocessing.d_ori_edge,
                                        d_edge = cfg.encoder.path_encoder.d_edge,
                                        expansion_factor = cfg.encoder.path_encoder.expansion_factor,
                                        d_relation = cfg.encoder.d_relation)
        
        self.edge_encoder = EdgeEncoder(edge_type_num = cfg.preprocessing.edge_type_num,
                                        d_ori_edge = cfg.preprocessing.d_ori_edge,
                                        d_edge = cfg.encoder.edge_encoder.d_edge,
                                        expansion_factor = cfg.encoder.edge_encoder.expansion_factor,
                                        d_relation = cfg.encoder.d_relation)

        if cfg.task == 'node_classification':
            self.genova_encoder_layers = nn.ModuleList([ \
                GenovaEncoderLayer(hidden_size = cfg.encoder.hidden_size,d_relation = cfg.encoder.d_relation,
                                   encoder_layer_num = cfg.encoder.num_layers)]*cfg.encoder.num_layers)
        else:
            self.genova_encoder_layers = nn.ModuleList([ \
                GenovaEncoderLayer(hidden_size = cfg.encoder.hidden_size,d_relation = cfg.encoder.d_relation,
                                   encoder_layer_num = cfg.encoder.num_layers,
                                   decoder_layer_num = cfg.decoder.num_layers)]*cfg.encoder.num_layers)

    def forward(self, node_input, path_input, edge_input, rel_mask):
        
        node = self.node_encoder(**node_input)
        edge = self.edge_encoder(**edge_input)
        path = self.path_encoder(**path_input)
        
        for genova_encoder_layer in self.genova_encoder_layers:
            node = genova_encoder_layer(node, edge, path, rel_mask)
        
        return node