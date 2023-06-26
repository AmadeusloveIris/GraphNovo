import os
import pickle
import gzip
import torch
import torch.nn.functional as F
import numpy as np
from .knapsack import knapsack_mask
from ..utils.BasicClass import Residual_seq, Ion, Composition

MS1_TOL = 5 # ppm
MS2_TOL = 0.02

class GenerationInference():
    def __init__(self, cfg, device, graphnovo_dir, spec_header, optimum_path_result, model, aa_id):
        self.beam_size = cfg.infer.beam_size
        self.data_dir = cfg.infer.data_dir
        self.device = device
        self.graphnovo_dir = graphnovo_dir
        self.spec_header = spec_header
        self.optimum_path_result = optimum_path_result
        self.aa_mass_dict = {aa: Residual_seq(aa).mass for aa in Residual_seq.output_aalist()}
        self.aa_id = aa_id 
        self.id_aa = {aa_id[aa]:aa for aa in aa_id} 
        self.aa_known_list = ['A', 'D', 'c', 'E', 'G', 'H', 'I', 'M', 'P', 'S', 'T', 'Y', 'V']
        self.aa_mass_min = min(self.aa_mass_dict.values())
        self.knapsack_matrix = np.load(os.path.join(self.graphnovo_dir,cfg.serialized_model_path.split('/')[0],'knapsack/knapsack.npy'))
        self.model = model

    def read_spec_data(self, idx):
        spec_head = self.spec_header.loc[idx[0]]
        seq_label = spec_head['Annotated Sequence'].replace('L', 'I')

        with open(os.path.join(self.graphnovo_dir, self.data_dir, spec_head['MSGP File Name']), 'rb') as f:
            f.seek(spec_head['MSGP Datablock Pointer'])
            spec = pickle.loads(gzip.decompress(f.read(spec_head['MSGP Datablock Length'])))
            node_mass = spec['node_mass']
            precursor_moverz = spec_head['m/z']
            precursor_charge = spec_head['Charge']
            pred_path = [int(p) for p in self.optimum_path_result.loc[idx[0]].pred_path.strip().split(' ')]
            path_mass = node_mass[pred_path]
            edge_known_list = []
            for p in self.optimum_path_result.loc[idx[0]].pred_seq.strip().split(' '):
                try:
                    edge_known_list.append(float(p))
                except:
                    edge_known_list.append(p)

        return seq_label, node_mass, path_mass, precursor_moverz, precursor_charge, edge_known_list

    def inference_step(self, spec_data, static_input, generate_next_token_prob, generate_model_input):
        
        node_mass, path_mass, precursor_moverz, precursor_charge, edge_known_list = spec_data
        seq_total_mass = sum([p for p in edge_known_list if isinstance(p, float)])

        with torch.no_grad():
            pred_seq = ''
            model_input = generate_model_input([pred_seq], node_mass, self.aa_id, \
                self.aa_mass_dict, self.input_cuda)
            tgt = generate_next_token_prob(self.model, model_input, static_input)
            tgt_list = [tgt[0][-1]]
            pred_seq_list = [pred_seq]
            score_list = [torch.tensor(0.).to(self.device)]
            edge_index = 0
            score_complete, pred_seq_list_complete = [], []
            pred_seq_list_final_complete, score_final_complete = [], []

            while True:
                # choose best k from seqs generated by beam search and the current pred_seq_complete 
                # and update current score_complete(removing the low prob)
                if isinstance(edge_known_list[edge_index], str):
                    pred_seq_list_tmp = []
                    score_list_tmp = []
                    aa = edge_known_list[edge_index]
                    for p_idx, p in enumerate(pred_seq_list):
                        pred_seq_list_tmp.append(p+aa)
                        score_list_tmp.append(score_list[p_idx]+F.log_softmax(tgt_list[p_idx], dim=-1)[self.aa_id[aa]])
                    pred_seq_list = pred_seq_list_tmp
                    score_list = score_list_tmp
                    assert len(pred_seq_list_complete) == 0
                    assert len(score_complete) == 0
                else:
                    pred_seq_list, score_list, score_complete, pred_seq_list_complete = \
                        self.beam_best_k(pred_seq_list, tgt_list, score_list, score_complete, \
                        pred_seq_list_complete, path_mass, seq_total_mass)

                # update all lists (moving completes from pred_seq_list into pred_seq_complete or final_complete)
                pred_seq_list_complete, score_complete, \
                pred_seq_list_incomplete, score_incomplete, \
                pred_seq_list_final_complete, score_final_complete = \
                    self.divide_complete_incomplete(pred_seq_list, score_list, pred_seq_list_complete, score_complete, \
                        pred_seq_list_final_complete, score_final_complete, path_mass[edge_index], path_mass[-1])
                
                if self.beam_size == len(score_final_complete) or len(score_incomplete)+len(score_complete) == 0:
                    pred_seq = self.pick_best_pred_seq(score_final_complete, pred_seq_list_final_complete, \
                                                       precursor_charge, precursor_moverz)
                    pred_prob = self.extract_pred_prob(static_input, pred_seq, node_mass, edge_known_list, path_mass, \
                                                       generate_model_input, generate_next_token_prob)
                    break
                elif self.beam_size == len(score_complete) or len(score_incomplete) == 0:
                    score_list, pred_seq_list = self.pick_best_k_pred_seq(\
                        score_complete, pred_seq_list_complete, path_mass[edge_index])
                    edge_index += 1
                    pred_seq_list_complete, score_complete = [], []
                else:
                    pred_seq_list = pred_seq_list_incomplete
                    score_list = score_incomplete
                
                pred_seq_list_incomplete = []
                score_incomplete = []

                tgt_list = []
                for pred_seq in pred_seq_list:
                    model_input = generate_model_input([pred_seq], node_mass, self.aa_id, \
                                  self.aa_mass_dict, self.input_cuda)
                    tgt = generate_next_token_prob(self.model, model_input, static_input)
                    tgt_list.append(tgt[0][-1])

        return pred_seq, pred_prob

    def beam_best_k(self, pred_seq_list, tgt_list, score_list, score_complete, pred_seq_list_complete, path_mass, seq_total_mass):
        """tgt_list: list of tgt[-1]"""
        aa_type_num = tgt_list[0].shape[0]
        total_single_aa_candidate = 0

        error_tolerance = round(10000*MS2_TOL + MS1_TOL*1e-2*path_mass[-1]) 
        while total_single_aa_candidate == 0:
            score_list_extend = []
            score_list_extend_concat = []
            total_single_aa_candidate = 0
            for t_idx, tgt in enumerate(tgt_list):
                # knapsack #
                if len(pred_seq_list[t_idx]) == 0:
                    sub_total_mass = 0
                else:
                    sub_total_mass = Residual_seq(pred_seq_list[t_idx]).mass

                node_idx = np.searchsorted(path_mass, sub_total_mass + 50)
                predict_mask, single_aa_candidate = knapsack_mask(tgt, self.knapsack_matrix, seq_total_mass, path_mass[node_idx]-sub_total_mass,
                                                                  error_tolerance, 10000, self.aa_id)
                
                if len(single_aa_candidate) == 0:
                    score_sum = torch.zeros_like(tgt)
                    score_sum = torch.where(score_sum.bool(), 0.0, -float('inf'))
                else:
                    score_sum = score_list[t_idx] + F.log_softmax(tgt, dim=-1) + predict_mask
                score_list_extend.append(score_sum)
                score_list_extend_concat.append(score_sum)
                # score_list_extend_concat.append(score_sum / (len(pred_seq_list[t_idx]) + 1))  # extend another token
                total_single_aa_candidate += len(single_aa_candidate)
            total_single_aa_candidate += len(score_complete)
            error_tolerance += 10000*MS2_TOL # this is for the exception that there is no candidate
        for s in score_complete:
            score_list_extend_concat.append(s.unsqueeze(0))

        score_list_extend_concat = torch.concat(score_list_extend_concat, dim=-1)
        beam_size = min(sum(score_list_extend_concat > float('-inf')), self.beam_size)
        beam_size = min(beam_size, total_single_aa_candidate)
        score_list_extend_concat = torch.nan_to_num(score_list_extend_concat, nan=-float('inf'))
        topk_index = torch.topk(score_list_extend_concat, k=beam_size)[-1]

        topk_pred_seq_list = []
        new_score_list = []
        new_score_complete = []
        new_pred_seq_list_complete = []

        for idx in topk_index:
            graph_index = int(int(idx) / aa_type_num)
            if graph_index < len(tgt_list):
                aa_index = idx % aa_type_num
                topk_pred_seq_list.append(pred_seq_list[graph_index] + self.id_aa[aa_index.item()])
                new_score_list.append(score_list_extend[graph_index][aa_index])
            else:
                score_index = idx - len(tgt_list) * aa_type_num
                new_score_complete.append(score_complete[score_index])
                new_pred_seq_list_complete.append(pred_seq_list_complete[score_index])
        return topk_pred_seq_list, new_score_list, new_score_complete, new_pred_seq_list_complete

    def divide_complete_incomplete(self, pred_seq_list, score_list, pred_seq_list_complete, score_complete, \
                        pred_seq_list_final_complete, score_final_complete, target_mass, precursor_mass):
        pred_seq_list_incomplete = []
        score_incomplete = []
        for p_idx, pred_seq in enumerate(pred_seq_list):
            sub_total_mass = Residual_seq(pred_seq).mass
            if sub_total_mass + 10 > precursor_mass or len(pred_seq) >= 32:
                pred_seq_list_final_complete.append(pred_seq)
                score_final_complete.append(score_list[p_idx])
                # score_final_complete.append(score_list[p_idx] / len(pred_seq))
            elif sub_total_mass + 10 > target_mass:
                pred_seq_list_complete.append(pred_seq)
                score_complete.append(score_list[p_idx])
                # score_complete.append(score_list[p_idx] / len(pred_seq))
            else:
                pred_seq_list_incomplete.append(pred_seq)
                score_incomplete.append(score_list[p_idx])

        return pred_seq_list_complete, score_complete, pred_seq_list_incomplete, score_incomplete, \
               pred_seq_list_final_complete, score_final_complete

    def pick_best_pred_seq(self, score_complete, pred_seq_list_complete, precursor_charge, precursor_moverz):
        score_complete = torch.concat([s.unsqueeze(0) for s in score_complete])
        seq_index_sort = torch.sort(score_complete, descending=True)[-1]
        mass_equal_flag = False
        for seq_index in seq_index_sort:
            pred_seq = pred_seq_list_complete[seq_index]
            theo = (Residual_seq(pred_seq).mass + Composition('H2O').mass + precursor_charge * Composition(
                'H').mass) / precursor_charge
            if abs(theo - precursor_moverz) < MS1_TOL * 1e-6 * theo:
                mass_equal_flag = True
                break
        if not mass_equal_flag:
            seq_index = torch.argmax(score_complete).item()
            pred_seq = pred_seq_list_complete[seq_index]

        return pred_seq

    def pick_best_k_pred_seq(self, score_complete, pred_seq_list_complete, target_mass):
        score_complete = torch.concat([s.unsqueeze(0) for s in score_complete])
        seq_index_sort = torch.sort(score_complete, descending=True)[-1]

        new_score_complete = []
        new_pred_seq_list_complete = []
        for seq_index in seq_index_sort:
            pred_seq = pred_seq_list_complete[seq_index]
            theo = Residual_seq(pred_seq).mass
            # if abs(theo - target_mass) < (MS2_TOL + MS1_TOL*1e-6 * precursor_ion_mass):
            new_score_complete.append(score_complete[seq_index])
            new_pred_seq_list_complete.append(pred_seq)
            if len(new_score_complete) == self.beam_size:
                break
        return new_score_complete, new_pred_seq_list_complete

    def input_cuda(self, input):
        if isinstance(input, torch.Tensor):
            input = input.to(self.device)
        else:
            for section_key in input:
                if isinstance(input[section_key], torch.Tensor):
                    input[section_key] = input[section_key].to(self.device)
                    continue
                for key in input[section_key]:
                    if isinstance(input[section_key][key], torch.Tensor):
                        input[section_key][key] = input[section_key][key].to(self.device)
        return input

    def match_AA_novor(cls, target, predicted):
        num_match = 0
        target_len = len(target)
        predicted_len = len(predicted)
        target_mass = np.array([Residual_seq(aa).mass for aa in target])
        target_mass_cum = np.concatenate([[0], np.cumsum(target_mass)])
        predicted_mass = np.array([Residual_seq(aa).mass for aa in predicted])
        predicted_mass_cum =  np.concatenate([[0], np.cumsum(predicted_mass)])

        i = 0
        j = 0
        while i < target_len and j < predicted_len:
            if abs(target_mass_cum[i] - predicted_mass_cum[j]) < 0.5:
                if abs(target_mass[i] - predicted_mass[j]) < 0.1:
                    num_match += 1
                i += 1
                j += 1
            elif target_mass_cum[i] < predicted_mass_cum[j]:
                i += 1
            else:
                j += 1
        return num_match, len(predicted), len(target)
    
    def extract_pred_prob(self, static_input, pred_seq, node_mass, edge_known_list, \
                        path_mass, generate_model_input, generate_next_token_prob):
        model_input = generate_model_input([pred_seq[:-1]], node_mass, self.aa_id,
                                            self.aa_mass_dict, self.input_cuda)
        tgt = generate_next_token_prob(self.model, model_input, static_input)
        tgt = F.softmax(tgt, dim=-1)
        pred_seq_id = [self.aa_id[aa] for aa in pred_seq]
        pred_prob = []
        edge_index = 0
        for t_idx, t in enumerate(tgt[0]):
            if isinstance(edge_known_list[edge_index], str):
                pred_prob.append(1)
                edge_index += 1
            else:
                pred_prob.append(t[pred_seq_id[t_idx]].item())
                if Residual_seq(pred_seq[:(t_idx+1)]).mass > path_mass[edge_index] - (MS2_TOL+MS1_TOL*1e-6*node_mass[-1]):
                    edge_index += 1
        return pred_prob
