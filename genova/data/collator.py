import torch
import numpy as np
from torch.nn.functional import pad

class GenovaCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        node_inputs = [record['node_input'] for record in batch]
        path_inputs = [record['rel_input'] for record in batch]
        edge_inputs = [record['edge_input'] for record in batch]
        node_labels = [record['graph_label'] for record in batch]
        
        node_shape = np.array([node_input['node_sourceion'].shape for node_input in node_inputs]).T
        max_node = node_shape[0].max()
        max_subgraph_node = node_shape[1].max()
        batch_num = len(batch)
        
        node_input = self.node_collate(node_inputs, max_node, max_subgraph_node)
        path_input = self.path_collate(path_inputs, max_node, node_shape)
        edge_input = self.edge_collate(edge_inputs, max_node)
        rel_mask = self.rel_collate(node_shape, max_node)
        node_labels, node_mask = self.nodelabel_collate(node_labels, max_node)
        
        encoder_input = {'node_input':node_input,'path_input':path_input,
                         'edge_input':edge_input,'rel_mask':rel_mask}
        labels = {'node_labels':node_labels, 'node_mask':node_mask}
        
        return encoder_input, labels

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