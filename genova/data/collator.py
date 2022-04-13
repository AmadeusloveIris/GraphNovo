import torch
import numpy as np
from torch.nn.functional import pad

class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        if self.cfg.task == 'optimum_path':
            spec = [record[0] for record in batch]
            tgt = [record[1] for record in batch]
            label = [record[2] for record in batch]
            encoder_input = self.encoder_collate(spec)
            decoder_input, graph_probability = self.decoder_collate(tgt)
            label, label_mask = self.label_collate(label)
            return encoder_input, decoder_input, graph_probability, label, label_mask
        
        elif self.cfg.task == 'sequence_generation':
            raise NotImplementedError
        
        elif self.cfg.task == 'node_classification':
            raise NotImplementedError
    
    def decoder_collate(self, decoder_input):
        if self.cfg.task == 'optimum_path':
            tgts_list = [record['tgt'] for record in decoder_input]
            trans_mask_list = [record['trans_mask'] for record in decoder_input]
            shape_list = np.array([tgt.shape for tgt in tgts_list])
            seqdblock_max = shape_list[:,0].max()
            node_max = shape_list[:,1].max()
            
            graph_probability = []
            trans_mask = []
            for i in range(len(tgts_list)):
                graph_probability.append(pad(tgts_list[i],[0,node_max-shape_list[i,1],
                                                           0,seqdblock_max-shape_list[i,0]]))
                trans_mask_temp = pad(trans_mask_list[i],[0,node_max-shape_list[i,1]],
                                      value=-float('inf'))
                trans_mask.append(pad(trans_mask_temp,[0,0,0,seqdblock_max-shape_list[i,0]]))
            graph_probability = torch.stack(graph_probability)
            decoder_input = {'trans_mask': torch.stack(trans_mask).unsqueeze(-1), 
                             'self_mask': (-float('inf')*torch.ones(seqdblock_max,seqdblock_max)) \
                             .triu(diagonal=1).unsqueeze(-1)}
            return decoder_input, graph_probability
            
    def label_collate(self, labels):
        if self.cfg.task == 'optimum_path':
            shape_list = np.array([label.shape for label in labels])
            seqdblock_max = shape_list[:,0].max()
            node_max = shape_list[:,1].max()
            result = []
            result_pading_mask = torch.ones(len(labels),seqdblock_max,dtype=bool)
            for i, label in enumerate(labels):
                result_pading_mask[i, label.shape[0]:] = 0
                label = pad(label,[0,node_max-label.shape[1],0,seqdblock_max-label.shape[0]])
                result.append(label)
            return torch.stack(result), result_pading_mask
    
    def encoder_collate(self, spec):
        node_inputs = [record['node_input'] for record in spec]
        path_inputs = [record['rel_input'] for record in spec]
        edge_inputs = [record['edge_input'] for record in spec]
        
        node_shape = np.array([node_input['node_sourceion'].shape for node_input in node_inputs]).T
        max_node = node_shape[0].max()
        max_subgraph_node = node_shape[1].max()
        
        node_input = self.node_collate(node_inputs, max_node, max_subgraph_node)
        path_input = self.path_collate(path_inputs, max_node, node_shape)
        edge_input = self.edge_collate(edge_inputs, max_node)
        rel_mask = self.rel_collate(node_shape, max_node)
        encoder_input = {'node_input':node_input,'path_input':path_input,
                         'edge_input':edge_input,'rel_mask':rel_mask}
        return encoder_input

    def node_collate(self, node_inputs, max_node, max_subgraph_node):
        node_feat = []
        node_sourceion = []
        charge = torch.IntTensor([node_input['charge'] for node_input in node_inputs])
        for node_input in node_inputs:
            node_num, node_subgraph_node = node_input['node_sourceion'].shape
            node_feat.append(pad(node_input['node_feat'], 
                                 [0, 0, 0, max_subgraph_node - node_subgraph_node, 0, max_node - node_num]))
            node_sourceion.append(pad(node_input['node_sourceion'], 
                                      [0, max_subgraph_node - node_subgraph_node, 0, max_node - node_num]))
        return {'node_feat':torch.stack(node_feat),'node_sourceion':torch.stack(node_sourceion),'charge':charge}
    
    def path_collate(self, path_inputs, max_node, node_shape):
        rel_type = torch.concat([path_input['rel_type'] for path_input in path_inputs]).squeeze(-1)
        rel_error = torch.concat([path_input['rel_error'] for path_input in path_inputs])
        rel_coor = torch.concat([pad(path_input['rel_coor'],[1,0],value=i) for i, path_input in enumerate(path_inputs)]).T
        rel_coor_cated = torch.stack([rel_coor[0]*max_node**2+rel_coor[1]*max_node+rel_coor[2],
                                      rel_coor[-2]*self.cfg.preprocessing.edge_type_num+rel_coor[-1]])
        
        rel_pos = torch.concat([path_input['rel_coor'][:,-2] for path_input in path_inputs])
        dist = torch.stack([pad(path_input['dist'],[0,max_node-node_shape[0,i],0,max_node-node_shape[0,i]]) for i, path_input in enumerate(path_inputs)])
        
        return {'rel_type':rel_type,'rel_error':rel_error,
                'rel_pos':rel_pos,'dist':dist,
                'rel_coor_cated':rel_coor_cated,
                'max_node': max_node, 'batch_num': len(path_inputs)}
        
        
    def edge_collate(self, edge_inputs, max_node):
        rel_type = torch.concat([edge_input['edge_type'] for edge_input in edge_inputs]).squeeze(-1)
        rel_error = torch.concat([edge_input['edge_error'] for edge_input in edge_inputs])
        rel_coor = torch.concat([pad(edge_input['edge_coor'],[1,0],value=i) for i, edge_input in enumerate(edge_inputs)]).T
        rel_coor_cated = torch.stack([rel_coor[0]*max_node**2+rel_coor[1]*max_node+rel_coor[2],
                                      rel_coor[-1]])
        
        return {'rel_type':rel_type,'rel_error':rel_error,
                'rel_coor_cated':rel_coor_cated, 
                'max_node': max_node, 'batch_num': len(edge_inputs)}
        
    def rel_collate(self, node_shape, max_node):
        rel_masks = []
        for i in node_shape[0]:
            rel_mask = -np.inf * torch.ones(max_node,max_node,1)
            rel_mask[:,:i] = 0
            rel_masks.append(rel_mask)
        rel_masks = torch.stack(rel_masks)
        return rel_masks
    
    def nodelabel_collate(self, node_labels_temp, max_node):
        node_mask = torch.ones(len(node_labels_temp),max_node).bool()
        node_labels = []
        for i, node_label in enumerate(node_labels_temp):
            node_mask[i, node_label.shape[0]:] = 0
            node_labels.append(pad(node_label,[0,max_node-node_label.shape[0]]))
        node_labels = torch.stack(node_labels)
        return node_labels, node_mask