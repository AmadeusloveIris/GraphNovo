import zmq
import sys
import gzip
import torch
import pickle
import numpy as np
import pandas as pd
from BasicClass import Residual_seq

def path_label_generator(seq, graphnode_moverz):
    start_index = graphnode_moverz.searchsorted(Residual_seq(seq).step_mass[:-1]-0.02)
    end_index = graphnode_moverz.searchsorted(Residual_seq(seq).step_mass[:-1]+0.02)
    path_label = np.zeros((graphnode_moverz.size,len(seq)))
    path_label[0, 0] = 1
    path_label[-1, -1]=1
    for i, (start, end) in enumerate(zip(start_index,end_index)): path_label[start: end, i]=1
    return path_label

if __name__=='__main__':
    context = zmq.Context()
    receive = context.socket(zmq.PULL)
    receive.connect('tcp://127.0.0.1:5558')
    worker = sys.argv[1]
    file_num = 1
    psm = pd.read_csv('/data/z37mao/genova/pretrain_data_sparse/genova_psm.csv',index_col=0)

    working_flag = True
    with open("/data/z37mao/genova/pretrain_data_sparse/temp_index/{}_index.csv".format(worker),'w',buffering=1) as index_writer:
        index_writer.write('index,Node Number,Serialized File Name,Serialized File Pointer,Serialized Data Length\n')
        while working_flag:
            with open("/data/z37mao/genova/pretrain_data_sparse/{}_{}.msgp".format(worker, file_num), 'wb') as writer:
                for i in range(8000):
                    if receive.poll(10000, zmq.POLLIN):
                        spec_index = receive.recv_string()
                        experiment, file_id, scan = spec_index.split('_')
                        temp = np.load('/home/z37mao/pretrain_data/{}/{}/{}.npz'.format(experiment, file_id, scan))
                        record = {}

                        #graph node data
                        graphnode_moverz = temp['graphnode_moverz']
                        graphnode_feature = temp['graphnode_feature'].astype(np.float32)
                        graphnode_feature[0,0,0]=1
                        graphnode_feature = np.concatenate((np.repeat(np.exp(-graphnode_moverz/3500).reshape(-1,1,1),graphnode_feature.shape[1],axis=1),graphnode_feature),axis=2)
                        graph_mask = np.any(graphnode_feature[:,:,1:]>0,axis=-1)
                        graphnode_iontype = np.where(graph_mask,graphnode_feature[:,:,-2]+1,
                                                    graphnode_feature[:,:,-2]).astype(np.uint8)
                        graphnode_feature = np.delete(graphnode_feature,-2,axis=-1)
                        graphnode_feature[:,:,0] = np.where(graph_mask,graphnode_feature[:,:,0],0)
                        record['node_feat'] = torch.Tensor(graphnode_feature)
                        record['node_sourceion'] = torch.IntTensor(graphnode_iontype)
                        
                        #graph relation data (sparse)
                        path_mask = temp['path_mask']
                        path_matrix = temp['path_matrix']
                        path_rdifferent_matrix = temp['path_rdifferent_matrix']
                        path_mmask = path_matrix>0
                        
                        rel_type = path_matrix[path_mmask]
                        rel_error = path_rdifferent_matrix[path_mmask]
                        dist = np.any(path_matrix>0,axis=-1).sum(axis=-1)
                        rel_coor = np.array(np.where(path_mmask))
                        rel_mask = path_mask+path_mask.T+np.eye(path_matrix.shape[0],dtype=bool)
                        rel_mask = np.where(rel_mask,0,-float('inf'))
                    
                        record['rel_type'] = torch.IntTensor(rel_type)
                        record['rel_error'] = torch.Tensor(rel_error)
                        record['edge_pos'] = torch.IntTensor(rel_coor[-2])
                        record['rel_coor'] = torch.LongTensor(rel_coor)
                        
                        #relation dense data
                        record['dist'] = torch.IntTensor(dist)
                        record['rel_mask'] = torch.Tensor(rel_mask)

                        #graph edge data
                        edge_mask = temp['adjacency_matrix']
                        edge_type = temp['edge_matrix']
                        edge_error = temp['edge_rdifferent_matrix']
                        
                        edge_mask += np.eye(edge_mask.shape[0],dtype=bool)
                        edge_mask = np.where(edge_mask==0,-float('inf'),0)

                        record['edge_type'] = torch.IntTensor(edge_type)
                        record['edge_error'] = torch.Tensor(edge_error)
                        record['edge_mask'] = torch.Tensor(edge_mask)
                        
                        #path label
                        path_label = path_label_generator(psm.loc[spec_index,'Annotated Sequence'].replace('L','I'), graphnode_moverz)
                        record['path_label'] = torch.Tensor(path_label)

                        #write file
                        compressed_data = gzip.compress(pickle.dumps(record))
                        index_writer.write('{},{},{},{},{}\n'.format(spec_index,graphnode_moverz.size,"{}_{}.msgp".format(worker, file_num),writer.tell(),len(compressed_data)))
                        writer.write(compressed_data)
                        
                    else:
                        working_flag = False
                        break
            file_num += 1
