import torch
import numpy as np
from torch.nn.functional import pad

class GenovaCollator(object):
    def __init__(self,cfg,*,mode):
        self.cfg = cfg
        self.mode = mode
        
    def __call__(self,batch):
        encoder_records = [record[0] for record in batch]
        encoder_input, node_mask = self.encoder_collate(encoder_records)
        if self.mode == 'train':
            if self.cfg.dataset.use_path_label:
                decoder_records = [record[1] for record in batch]
                labels = [record[2] for record in batch]
                labels = self.path_label_pad(labels)
                decoder_input = self.decoder_collate_path(decoder_records)
                decoder_input['memory_key_padding_mask'] = node_mask
                return encoder_input, decoder_input, labels 
            else:
                seqs = [record[1] for record in batch]
                tf_labels, labels = self.seq_pad(seqs)
                decoder_input = {}
                decoder_input['tgt_index'] = tf_labels
                decoder_input['memory_key_padding_mask'] = node_mask
                decoder_input['tgt_mask'] = torch.triu(torch.full((tf_labels.shape[1], tf_labels.shape[1]), float('-inf')), diagonal=1)
                return encoder_input, decoder_input, labels
        else:
            if self.cfg.dataset.use_path_label:
                raise NotImplementedError
            else:
                seqs = [record[1] for record in batch]
                tf_labels, labels = self.seq_pad(seqs)
                decoder_input = {}
                decoder_input['tgt_index'] = tf_labels
                decoder_input['memory_key_padding_mask'] = node_mask
                return encoder_input, node_mask, labels
        
    def encoder_collate(self, encoder_records):
        node_shape = []
        for record in encoder_records: node_shape.append(np.array(record['node_sourceion'].shape))
        node_shape = np.array(node_shape).T
        max_node = node_shape[0].max()
        max_subgraph_node = node_shape[1].max()

        node_input = {}
        edge_input = {}
        rel_input = {}

        edge_input['rel_type'] = torch.concat([record['rel_type'] for record in encoder_records])
        edge_input['edge_pos'] = torch.concat([record['edge_pos'] for record in encoder_records])
        edge_input['rel_error'] = torch.concat([record['rel_error'] for record in encoder_records]).unsqueeze(-1)


        node_feat = []
        node_sourceion = []
        rel_mask = []
        dist = []
        charge = []
        rel_coor_cated = []
        node_mask = torch.zeros(len(encoder_records),max_node,dtype=bool)
        for i, record in enumerate(encoder_records):
            node_num, node_subgraph_node = record['node_sourceion'].shape
            node_feat.append(pad(record['node_feat'],[0,0,0,max_subgraph_node-node_subgraph_node,0,max_node-node_num]))
            node_sourceion.append(pad(record['node_sourceion'],[0,max_subgraph_node-node_subgraph_node,0,max_node-node_num]))
            rel_mask.append(pad(pad(record['rel_mask'],[0,max_node-node_num],value=-float('inf')),[0,0,0,max_node-node_num]))
            dist.append(pad(record['dist'],[0,max_node-node_num,0,max_node-node_num]))
            charge.append(record['charge'])
            rel_coor_cated.append(torch.stack([i*max_node**2+record['rel_coor'][0]*max_node+record['rel_coor'][1],
                                               record['rel_coor'][-2]*100+record['rel_coor'][-1]]))
            node_mask[i,node_num:] = True

        drctn = torch.zeros(max_node,max_node)+torch.tril(2*torch.ones(max_node,max_node),-1)+torch.triu(torch.ones(max_node,max_node),1)
        rel_input['drctn'] = drctn.int().unsqueeze(0)
        node_input['node_feat'] = torch.stack(node_feat)
        node_input['node_sourceion'] = torch.stack(node_sourceion)
        rel_input['rel_mask'] = torch.stack(rel_mask).unsqueeze(-1)
        edge_input['dist'] = torch.stack(dist)
        node_input['charge'] = torch.IntTensor(charge)
        edge_input['rel_coor_cated'] = torch.concat(rel_coor_cated,dim=1)
        edge_input['batch_num'] = len(encoder_records)
        edge_input['max_node'] = max_node
        
        encoder_input = {'node_input':node_input,'edge_input':edge_input,'rel_input':rel_input}

        return encoder_input, node_mask

    def decoder_collate_path(self, decoder_records):
        decoder_input = {}
        node_num = [record['edge_attn_mask'].size(1) for record in decoder_records]
        max_node = max(node_num)
        seq_len = [record['edge_attn_mask'].size(0) for record in decoder_records]
        max_seq_len = max(seq_len)

        decoder_input['edge_type'] = torch.concat([record['edge_type'] for record in decoder_records])
        decoder_input['edge_error'] = torch.concat([record['edge_error'] for record in decoder_records]).unsqueeze(1)

        edge_attn_mask = []
        edge_label_mask = []
        edge_coor_cated = []
        path_label = []
        for i, record in enumerate(decoder_records):
            edge_coor_cated.append(torch.stack([i*max_node*max_seq_len+record['edge_coor'][0]*max_node+record['edge_coor'][1],
                                                record['edge_coor'][-1]]))
            edge_attn_mask.append(pad(pad(record['edge_attn_mask'],(0,max_node-node_num[i]),value=-float('inf')),(0,0,0,max_seq_len-seq_len[i])))
            edge_label_mask.append(pad(pad(record['edge_label_mask'],(0,max_node-node_num[i]),value=-float('inf')),(0,0,0,max_seq_len-seq_len[i])))
            path_label.append(pad(record['path_label'],(0,max_node-node_num[i],0,max_seq_len-seq_len[i])))
            
        decoder_input['max_node'] = max_node
        decoder_input['max_seq_len'] = max_seq_len
        decoder_input['edge_coor_cated'] = torch.concat(edge_coor_cated,dim=1)
        decoder_input['edge_attn_mask'] = torch.stack(edge_attn_mask)
        decoder_input['edge_label_mask'] = torch.stack(edge_label_mask)
        decoder_input['path_label'] = torch.stack(path_label)
        return decoder_input

    def seq_pad(self, seqs):
        seq_len = np.array([seq.size(0) for seq in seqs])-1
        max_seq_len = max(seq_len)
        tf_labels = torch.stack([pad(seq[:-1],(0,max_seq_len - seq_len[i])) for i, seq in enumerate(seqs)])
        labels = torch.stack([pad(seq[1:],(0,max_seq_len - seq_len[i])) for i, seq in enumerate(seqs)])
        return tf_labels, labels

    def path_label_pad(self, path_labels):
        node_num = [path_label.size(1) for path_label in path_labels]
        max_node = max(node_num)
        seq_len = [path_label.size(0) for path_label in path_labels]
        max_seq_len = max(seq_len)

        labels = torch.stack([pad(path_label,(0,max_node-node_num[i],0,
                                  max_seq_len-seq_len[i])) for i, path_label in enumerate(path_labels)])
        return labels