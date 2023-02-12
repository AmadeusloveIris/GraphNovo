import torch
from genova.utils.BasicClass import Residual_seq, Ion
from itertools import combinations_with_replacement
        
def aa_dict_generate():
    aa_dict = {}
    aalist = Residual_seq.output_aalist()
    for aa in aalist:
        aa_dict[aa] = Residual_seq(aa).mass
    return aa_dict

def aa_datablock_dict_generate():
    aa_datablock_dict = {}
    aa_datablock_dict_tmp = {}
    aa_datablock_dict_reverse = {}
    aalist = Residual_seq.output_aalist()
    for num in range(1, 7):
        for i in combinations_with_replacement(aalist, num):
            aa_datablock_dict_tmp[i] = Residual_seq(i).mass

    sorted_aa_datablock_dict = sorted(aa_datablock_dict_tmp.items(), key=lambda kv: kv[1])
    last_v = sorted_aa_datablock_dict[0][1]
    same_v_klist = [last_v]
    last_k = sorted_aa_datablock_dict[0][0]

    for i in range(1, len(sorted_aa_datablock_dict)):
        k, v = sorted_aa_datablock_dict[i]

        if v == last_v:
            same_v_klist.append(k)
        else:
            if len(same_v_klist) == 1:
                aa_datablock_dict[last_k] = last_v
                aa_datablock_dict_reverse[last_v] = last_k
            elif len(same_v_klist) > 1:
                for j, sv_k in enumerate(same_v_klist):
                    aa_datablock_dict[sv_k] = last_v + 0.000000000001 * j
                    aa_datablock_dict_reverse[last_v + 0.000000000001 * j] = sv_k
            last_v = v
            last_k = k
            same_v_klist = [last_k]

    if len(same_v_klist) == 1:
        aa_datablock_dict[last_k] = last_v
        aa_datablock_dict_reverse[last_v] = last_k
    elif len(same_v_klist) > 1:
        for j, sv_k in enumerate(same_v_klist):
            aa_datablock_dict[sv_k] = last_v + 0.000000000001 * j
            aa_datablock_dict_reverse[last_v + 0.000000000001 * j] = sv_k

    return aa_datablock_dict, aa_datablock_dict_reverse

def input_cuda(input, device):
    for section_key in input:
        if isinstance(input[section_key], torch.Tensor):
            input[section_key] = input[section_key].to(device)
            continue
        for key in input[section_key]:
             if isinstance(input[section_key][key], torch.Tensor):
                input[section_key][key] = input[section_key][key].to(device)
    return input
