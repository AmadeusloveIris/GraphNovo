import os
import csv
import pickle
import gzip
import torch
import torch.nn.functional as F
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from .infer_utils import aa_datablock_dict_generate, input_cuda
from ..utils.BasicClass import Residual_seq, Ion

def trans_mask_generate(node_mass, graph_label, dist, aa_mass_dict):
    """ For single sample """
    node_num = node_mass.size
    edge_mask = torch.zeros(node_num,node_num,dtype=bool)
    for x,y in enumerate(node_mass.searchsorted(node_mass+min(aa_mass_dict.values())*7+0.04)):
        edge_mask[x,y:] = True
    edge_mask = torch.logical_or(edge_mask,dist!=0)
    trans_mask=((graph_label@edge_mask.int())!=0).bool()
    trans_mask = torch.where(trans_mask,0.0,-float('inf'))
    return trans_mask


def generate_decoder_input(time_step, graph_probability_input_list, node_mass, dist, aa_mass_dict, device):
    """ For single sample due to node_mass and dist"""
    decoder_input = {}
    decoder_input['self_mask'] = (-float('inf') * torch.ones(time_step, time_step)).triu(diagonal=1).unsqueeze(-1)
    predict_graph_labels = (graph_probability_input_list.cpu() > 0).int()
    decoder_input['trans_mask'] = []
    for predict_graph_label in predict_graph_labels:
        decoder_input['trans_mask'].append(trans_mask_generate(node_mass, predict_graph_label, dist,
                                                               aa_mass_dict).unsqueeze(-1))
    decoder_input['trans_mask'] = torch.stack(decoder_input['trans_mask'], dim=0)
    decoder_input = input_cuda(decoder_input, device)

    return decoder_input


def generate_graph_probability(graph_probability_input, decoder_input, graph_node, model):
    """ For the batch """
    query_node = graph_probability_input @ graph_node
    query_node = model.decoder(**decoder_input, tgt=query_node, graph_node=graph_node)
    query_node = model.query_node_linear(query_node)
    graph_node_t = model.graph_node_linear(graph_node).transpose(1, 2)
    graph_probability = query_node @ graph_node_t
    graph_probability = graph_probability + decoder_input['trans_mask'].squeeze(-1)

    return graph_probability

def extract_pred_prob(graph_node, graph_probability_input, node_mass, dist, aa_mass_dict, device, model):
    graph_probability = graph_probability_input[0,1:,:]
    graph_probability_input = graph_probability_input[:,:-1,:]
    decoder_input = generate_decoder_input(graph_probability_input.shape[1], \
                    graph_probability_input, node_mass, dist, aa_mass_dict, device)
    pred_prob = generate_graph_probability(graph_probability_input, decoder_input, graph_node, model)
    pred_prob = pred_prob[0].softmax(-1)
    pred_prob = pred_prob * graph_probability
    pred_prob = pred_prob[pred_prob>0].cpu().tolist()
    return pred_prob

def beam_best_k(graph_probability_input_list, graph_probability_list, score_list,
                score_complete, graph_probability_input_complete, beam_size, sum_flag, device):
    """ Return k best results and the corresponding scores.
        For single sample.
    """
    path_len, node_num = graph_probability_input_list[0].shape

    score_list_extend = []
    for g_idx, graph_probability in enumerate(graph_probability_list):
        sum_score = score_list[g_idx] + F.log_softmax(graph_probability[-1], dim=-1)
        score_list_extend.append(sum_score)

    score_list_extend_concat = None
    if sum_flag == 'sum':
        score_list_extend_concat = score_list_extend
    elif sum_flag == 'average':
        score_list_extend_concat = [s/(path_len+1) for s in score_list_extend] # extend another token
    
    for s in score_complete:
        score_list_extend_concat.append(s.unsqueeze(0))

    score_list_extend_concat = torch.concat(score_list_extend_concat, dim=-1)
    topk_index = torch.topk(score_list_extend_concat, k=beam_size)[-1]

    topk_graph_probability_input = []
    new_score_list = []
    new_score_complete = []
    new_graph_probability_input_complete = []
    for idx in topk_index:
        graph_index = int(int(idx) / node_num)
        if graph_index < len(graph_probability_list):
            node_index = idx % node_num
            graph_probability_extend = torch.zeros(1, node_num).to(device)
            graph_probability_input = torch.concat((graph_probability_input_list[graph_index], graph_probability_extend),
                                                   dim=-2)
            graph_probability_input[-1, node_index] = 1
            topk_graph_probability_input.append(graph_probability_input)
            new_score_list.append(score_list_extend[graph_index][node_index])
        else:
            score_index = idx - len(graph_probability_list) * node_num
            new_score_complete.append(score_complete[score_index])
            new_graph_probability_input_complete.append(graph_probability_input_complete[score_index])

    return topk_graph_probability_input, new_score_list, new_score_complete, new_graph_probability_input_complete


def path_evaluation(graph_probability_input, label):
    """ For single sample """
    graph_probability_input = graph_probability_input[0][1:]
    label = label[0]

    labels_which_step, labels_pos_block = (label > 0).nonzero(as_tuple=True)
    pred_which_step, pred_pos_block = (graph_probability_input > 0).nonzero(as_tuple=True)

    pred_path = []
    prev_p = -1
    for p_idx, p in enumerate(pred_which_step):
        if p == prev_p:
            continue
        else:
            pred_path.append(pred_pos_block[p_idx])
            prev_p = p
    pred_path = torch.Tensor(pred_path).int()
    label_path = [[] for _ in range(label.shape[0])]
    for idx_l, l in enumerate(labels_which_step.cpu()):
        label_path[l].append(labels_pos_block[idx_l].cpu().item())

    matched_pos = np.intersect1d(labels_pos_block.cpu().numpy(), pred_path.cpu().numpy())
    matched_num = matched_pos.shape[0]

    which_steps = [labels_which_step[torch.where(labels_pos_block == m)].item() for m in matched_pos]
    matched_num_fix = len(which_steps) - len(set(which_steps))
    if matched_num_fix > 0:
        print('fixing matched_num_fix')
    matched_num -= matched_num_fix

    return matched_num, graph_probability_input.shape[0], label.shape[0], pred_path.tolist(), label_path

def format_seq_predict(predict_path, node_mass, aa_datablock, aa_datablock_dict_reverse, precursor_ion_mass):
    seq_predict = []
    path_mass = node_mass[[0]+predict_path]
    edge_mass_list = [path_mass[i + 1] - path_mass[i] for i in range(len(path_mass) - 1)]

    for edge_mass in edge_mass_list:
        mass_threshold = 2*0.02+5*precursor_ion_mass*1e-6
        # mass_diff-mass_threshold <= aa_l <= edge_mass+mass_threshold <= aa_r
        l = aa_datablock.searchsorted(edge_mass - mass_threshold, side='left')
        r = aa_datablock.searchsorted(edge_mass + mass_threshold, side='left')
        aa_values = [aa_datablock[idx] for idx in range(l, r)]

        # Note: here may have some states with no aa_block existing 
        aa_block = [''.join(aa_datablock_dict_reverse[aa_value]) for aa_value in aa_values]
        if len(aa_block) == 0 or len(aa_block) > 1 or (len(aa_block) == 1 and len(aa_block[0]) > 1):
            seq_predict.append(str(edge_mass))
        elif len(aa_block) == 1 and len(aa_block[0]) == 1:
            seq_predict.append(aa_block[0])
            assert edge_mass>Residual_seq(aa_block[0]).mass-mass_threshold
            assert edge_mass<Residual_seq(aa_block[0]).mass+mass_threshold
            assert aa_block[0] not in ['N', 'Q']

    seq_predict = ' '.join(seq_predict)

    return seq_predict
    
def optimal_path_infer(cfg:DictConfig, spec_header, test_dl, model, device):
    genova_dir = get_original_cwd()

    # dictionary
    aa_datablock_dict, aa_datablock_dict_reverse = aa_datablock_dict_generate()
    aa_datablock = np.array(sorted(list(aa_datablock_dict.values())))
    aa_mass_dict = {aa:Residual_seq(aa).mass for aa in Residual_seq.output_aalist()}
    
    # save result
    path_file_name = genova_dir + cfg.infer.optimal_path_file
    path_csvfile = open(path_file_name, 'w', buffering=1)
    path_fieldnames = ['graph_idx', 'pred_path', 'pred_prob', 'label_path', 'pred_seq']
    writer_path = csv.DictWriter(path_csvfile, fieldnames=path_fieldnames, quoting=csv.QUOTE_NONNUMERIC)
    writer_path.writeheader()

    # metrics initialization
    matched_num_total, predict_len_total, label_len_total = 0, 0, 0
    # main
    sum_flag = 'sum'
    for i, (encoder_input, _, _, label_path_distribution, _, idx) in enumerate(test_dl):
        if i % 100 == 0 and i > 0:
            print('Num of Samples: ', i)
        if torch.is_tensor(idx): idx = idx.tolist()
        spec_head = spec_header.loc[idx[0]]
        
        with open(os.path.join(genova_dir+cfg.infer.data_dir, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
            node_mass = spec['node_mass']
            dist = spec['rel_input']['dist']
            tensor_coor = torch.transpose(torch.unique(spec['edge_input']['edge_coor'][:,:2],dim=0), 0, 1)
            tensor_values = torch.ones(tensor_coor.shape[1])
            sparse_tensor = torch.sparse_coo_tensor(tensor_coor, tensor_values, dist.shape)
            dist = sparse_tensor.to_dense()
            assert dist.sum() == tensor_coor.shape[1]
            precursor_moverz = spec_head['m/z']
            precursor_charge = spec_head['Charge']
            precursor_ion_mass = Ion.precursorion2mass(precursor_moverz, precursor_charge)

        with torch.no_grad():
            time_step = 1
            encoder_input = input_cuda(encoder_input, device)
            graph_node = model.encoder(**encoder_input)
            _, node_num, _, _ = encoder_input['node_input']['node_feat'].shape

            # graph_probability_input & decoder_input for first time_step
            graph_probability_input = torch.zeros(1, 1, node_num)  # [batch, seq_len, node_num]
            graph_probability_input[:, :, 0] = 1
            graph_probability_input = graph_probability_input.to(device)
            decoder_input = generate_decoder_input(time_step, graph_probability_input, node_mass, dist, aa_mass_dict, device)
            graph_probability = generate_graph_probability(graph_probability_input, decoder_input, graph_node, model)

            # *_incomplete is for check the incomplete paths for every time step
            # *_complete is for storing the complete paths' information
            beam_size = min(cfg.infer.beam_size, spec_head['Node Number']-1)
            graph_probability_input_list = [graph_probability_input[0]]
            graph_probability_list = [graph_probability[0]]
            graph_probability_input_complete = []
            graph_probability_input_incomplete = []
            score_list = [0]
            score_complete = []
            score_incomplete = []

            while True:
                time_step += 1
                graph_probability_input_list, score_list, \
                score_complete, graph_probability_input_complete = \
                    beam_best_k(graph_probability_input_list, graph_probability_list, score_list,
                                score_complete, graph_probability_input_complete, beam_size, sum_flag, device)

                # if beam size is large enough, it is possible to have the final node as the next node on second
                # time step, then graph_probability_list[g_idx] will out of index.
                # check if any path from the beam has reached the final node and the incomplete paths will continue
                for g_idx, graph_probability_input in enumerate(graph_probability_input_list):
                    if graph_probability_input[-1][-1].item() > 0:
                        graph_probability_input_complete.append(graph_probability_input)
                        if sum_flag == 'sum':
                            score_complete.append(score_list[g_idx])
                        elif sum_flag == 'average':
                            score_complete.append(score_list[g_idx] / graph_probability_input.shape[0])
                    else:
                        graph_probability_input_incomplete.append(graph_probability_input)
                        score_incomplete.append(score_list[g_idx])

                # if the number of complete paths equal to beam size, it means the end to search
                if beam_size == len(score_complete):
                    score_complete = torch.concat([s.unsqueeze(0) for s in score_complete])
                    path_index = torch.argmax(score_complete).item()
                    graph_probability_input = graph_probability_input_complete[path_index].unsqueeze(0)
                    pred_prob = extract_pred_prob(graph_node[:1,:,:], graph_probability_input, node_mass, \
                                                dist, aa_mass_dict, device, model)
                    graph_probability_input = graph_probability_input.cpu()
                    break

                graph_probability_input_list = graph_probability_input_incomplete
                graph_probability_input_incomplete = []
                score_list = score_incomplete
                score_incomplete = []

                # all paths not complete continue to generate the distribution to choose next node of the path
                graph_probability_input_list = torch.stack(graph_probability_input_list, dim=0)
                decoder_input = generate_decoder_input(time_step, graph_probability_input_list, node_mass, dist, aa_mass_dict, device)
                graph_node = torch.stack(len(graph_probability_input_list) * [graph_node[0]], dim=0)
                graph_probability_list = generate_graph_probability(graph_probability_input_list, decoder_input, graph_node, model)

        # Do evaluation on optimal path for each sample
        matched_num, predict_len, label_len, pred_path, label_path = path_evaluation(graph_probability_input, label_path_distribution)
        path_pred_print = ' '.join([str(p) for p in pred_path])
        seq_predict = format_seq_predict(pred_path, node_mass, aa_datablock,
                                        aa_datablock_dict_reverse, precursor_ion_mass)

        path_label_print_tmp = ['/'.join([str(p) for p in ps]) for ps in label_path]
        path_label_print = ' '.join(path_label_print_tmp)
        matched_num_total += matched_num
        predict_len_total += predict_len
        label_len_total += label_len
        
        pred_prob = ' '.join([str(p) for p in pred_prob])
        writer_path.writerow({'graph_idx': idx[0], 'pred_path': path_pred_print, 'pred_prob': pred_prob,\
            'label_path': path_label_print, 'pred_seq':seq_predict})


    # Print the final evaluation
    print('matched_num_total: ', matched_num_total)
    print('predict_len_total: ', predict_len_total)
    print('label_len_total: ', label_len_total)
    print('path precision: ', matched_num_total / predict_len_total)
    print('path recall: ', matched_num_total / label_len_total)
