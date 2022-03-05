import zmq
import sys
import gzip
import torch
import pickle
import numpy as np
import pandas as pd
from BasicClass import Residual_seq

if __name__=='__main__':
    context = zmq.Context()
    receive = context.socket(zmq.PULL)
    receive.connect('tcp://127.0.0.1:5558')
    worker = sys.argv[1]

    working_flag = True
    with open("/data/z37mao/genova/test/{}_index.csv".format(worker),'w',buffering=1) as index_writer:
        index_writer.write('index,path_matrix_dense_num\n')
        while working_flag:
            for i in range(8000):
                if receive.poll(10000, zmq.POLLIN):
                    spec_index = receive.recv_string()
                    experiment, file_id, scan = spec_index.split('_')
                    temp = np.load('/home/z37mao/pretrain_data/{}/{}/{}.npz'.format(experiment, file_id, scan))
                    index_writer.write('{},{}\n'.format(spec_index,np.sum(temp['path_matrix']>0)))
                else:
                    working_flag = False
                    break
