import csv
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from .edge_version_inference import GenerationInference
from ..utils.BasicClass import Residual_seq

def generate_model_input(pred_seq_list, node_mass, aa_id, aa_mass_dict, input_cuda):
    tgt_input = {}
    for pred_seq in pred_seq_list:
        if len(pred_seq) == 0:
            tgt_input_seq = torch.Tensor([aa_id['<bos>']]).long()
        else:
            tgt_input_seq = torch.Tensor([aa_id['<bos>']] + [aa_id[aa] for aa in pred_seq]).long()
        seq_len = len(tgt_input_seq)
        tgt_input['seq'] = torch.stack([tgt_input_seq], dim=0)
        tgt_input['pos'] = torch.Tensor(list(range(seq_len))).long()
    input_cuda(tgt_input)

    decoder_input = {}
    decoder_input['self_mask'] = (-float('inf') * torch.ones(seq_len, seq_len)).triu(diagonal=1).unsqueeze(-1)
    decoder_input['trans_mask'] = []
    for pred_seq in pred_seq_list:
        decoder_input['trans_mask'].append(trans_mask_generate(pred_seq, node_mass, aa_mass_dict).unsqueeze(-1))
    decoder_input['trans_mask'] = torch.stack(decoder_input['trans_mask'], dim=0)
    input_cuda(decoder_input)
    return [tgt_input, decoder_input]

def trans_mask_generate(seq, node_mass, aa_mass_dict):
    seq_mass = np.array([0]+[aa_mass_dict[aa] for aa in seq]).cumsum()
    trans_mask = torch.zeros((seq_mass.size,node_mass.size))
    trans_mask[0,0] = -float('inf')
    for i, board in enumerate(node_mass.searchsorted(seq_mass+min(aa_mass_dict.values())-0.02),start=0):
        trans_mask[i,:board] = -float('inf')
    trans_mask[i, -1] = 0. 
    return trans_mask
    
def generate_next_token_prob(model, model_input, static_input):
    tgt_input, decoder_input = model_input
    tgt = model.tgt_embedding(tgt_input['seq']) + model.pos_embedding(tgt_input['pos'])
    tgt = model.decoder(**decoder_input, tgt=tgt, graph_node=static_input)
    tgt = model.output_linear(tgt)

    tgt_mask = torch.zeros_like(tgt)
    tgt_mask[:,:,:3] = float('-inf')
    tgt = tgt + tgt_mask
    return tgt
    

def seq_generation_infer(cfg: DictConfig, spec_header, test_dl, model, device):
    graphnovo_dir = get_original_cwd()
    optimal_path_result = pd.read_csv(os.path.join(graphnovo_dir, cfg.infer.optimal_path_file), index_col="graph_idx")
    optimal_path_result = optimal_path_result.drop(["label_path"], axis=1)

    # dictionary
    aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=3)}
    aa_dict['<pad>'] = 0
    aa_dict['<bos>'] = 1
    aa_dict['<eos>'] = 2
    
    # save result
    print('graphnovo_dir:', graphnovo_dir)
    filename = os.path.join(graphnovo_dir, cfg.infer.output_file)
    print("output file: ", filename)
    if os.path.isfile(cfg.infer.output_file):
        csvfile = open(filename, 'a', buffering=1)
        fieldnames = ['graph_idx', 'pred_seq', 'pred_prob', 'pred_path', 'label_seq']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    else:
        csvfile = open(filename, 'w', buffering=1)
        fieldnames = ['graph_idx', 'pred_seq', 'pred_prob', 'pred_path', 'label_seq']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
    
    aa_matched_num_total, aa_predict_len_total, aa_label_len_total = 0, 0, 0
    peptide_matched_num = 0
    gen_infer = GenerationInference(cfg, device, graphnovo_dir, spec_header,\
        optimal_path_result, model, aa_dict)
    for i, (encoder_input, _, _, _, _, idx) in enumerate(test_dl):
        if i % 100 == 0 and i > 0:
            print('Num of Samples: ', i)
        if torch.is_tensor(idx): idx = idx.tolist()

        seq_label, node_mass, path_mass, precursor_moverz,\
            precursor_charge, edge_known_list = gen_infer.read_spec_data(idx)
        seq_label_sep = seq_label
        
        try:
            with torch.no_grad():
                encoder_input = gen_infer.input_cuda(encoder_input)
                encoder_output = model.encoder(**encoder_input)
            
            pred_seq, pred_prob = gen_infer.inference_step([node_mass, path_mass, \
                precursor_moverz, precursor_charge, edge_known_list], \
                encoder_output, generate_next_token_prob, generate_model_input)

            writer.writerow({'graph_idx': idx[0], 'pred_seq': pred_seq, 'pred_prob': pred_prob, \
                            'pred_path':path_mass.tolist(), 'label_seq': seq_label_sep})
        except RuntimeError as e:
            if 'out of memory' in str(e): print(f'WARNING: {idx[0]} ran out of memory. Please run it on device with enough memory')
            pred_seq = ''
            pred_prob = ''
            
        aa_matched_num, aa_predict_len, aa_label_len = \
            gen_infer.match_AA_novor(seq_label.replace(' ',''), pred_seq)
        if aa_matched_num == aa_predict_len and aa_predict_len == aa_label_len:
            peptide_matched_num += 1
        aa_matched_num_total += aa_matched_num
        aa_predict_len_total += aa_predict_len
        aa_label_len_total += aa_label_len


    print('aa_matched_num_total:', aa_matched_num_total)
    print('aa_predict_len_total: ', aa_predict_len_total)
    print('aa_label_len_total: ', aa_label_len_total)
    print('aa precision: ', aa_matched_num_total / aa_predict_len_total)
    print('aa recall: ', aa_matched_num_total / aa_label_len_total)
    print('peptide recall: ', peptide_matched_num / spec_header.shape[0])
