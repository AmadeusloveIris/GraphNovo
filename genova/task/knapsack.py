import numpy as np
import torch
from genova.utils.BasicClass import Residual_seq

mass_AA_min = min([Residual_seq(aa).mass for aa in Residual_seq.output_aalist()])
# print(aa_dict)
# print(aa_dict_reverse)

def knapsack_build(mass_max, aa_resolution):
  """build knapsack matrix."""

  aa_dict = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=1)}
  aa_dict_reverse = {v: k for k, v in aa_dict.items()}

  vocab_size = len(Residual_seq.output_aalist())
  peptide_mass = mass_max
  # peptide_mass = peptide_mass - (deepnovo_config.mass_C_terminus + deepnovo_config.mass_H)
  print("peptide_mass = ", peptide_mass)

  peptide_mass_round = int(round(peptide_mass * aa_resolution))
  print("peptide_mass_round = ", peptide_mass_round)

  peptide_mass_upperbound = (peptide_mass_round + aa_resolution)
  knapsack_matrix = np.zeros(shape=(vocab_size+1, peptide_mass_upperbound),
                             dtype=bool)

  for aa_id in range(1, vocab_size+1):

    mass_aa_round = int(round(Residual_seq(aa_dict_reverse[aa_id]).mass * aa_resolution))
    print(aa_dict_reverse[aa_id], mass_aa_round)

    for col in range(peptide_mass_upperbound):

      # col 0 ~ mass 1
      # col + 1 = mass
      # col = mass - 1
      current_mass = col + 1

      if current_mass < mass_aa_round:
        knapsack_matrix[aa_id, col] = False

      if current_mass == mass_aa_round:
        knapsack_matrix[aa_id, col] = True

      if current_mass > mass_aa_round:
        sub_mass = current_mass - mass_aa_round
        sub_col = sub_mass - 1
        if np.sum(knapsack_matrix[:, sub_col]) > 0:
          knapsack_matrix[aa_id, col] = True
          knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col],
                                                  knapsack_matrix[:, sub_col])
        else:
          knapsack_matrix[aa_id, col] = False

  np.save("knapsack.npy", knapsack_matrix)


def knapsack_search(knapsack_matrix, peptide_mass, mass_precision_tolerance, aa_resolution, aa_dict):
  aa_dict_standard = {aa: i for i, aa in enumerate(Residual_seq.output_aalist(), start=1)}
  aa_dict_offset = aa_dict['A'] - aa_dict_standard['A']
  for aa in Residual_seq.output_aalist():
    assert aa_dict[aa] - aa_dict_standard[aa] == aa_dict_offset
  mass_AA_min_round = int(round(mass_AA_min * aa_resolution))
  peptide_mass_round = int(round(peptide_mass * aa_resolution))
  # 100 x 0.0001 Da
  peptide_mass_upperbound = peptide_mass_round + mass_precision_tolerance
  peptide_mass_lowerbound = peptide_mass_round - mass_precision_tolerance

  if peptide_mass_upperbound < mass_AA_min_round:
    return []

  if peptide_mass_lowerbound > knapsack_matrix.shape[1]:
    return [aa_dict[aa] for aa in Residual_seq.output_aalist()]

  # col 0 ~ mass 1
  # col + 1 = mass
  # col = mass - 1
  # [)
  peptide_mass_lowerbound_col = peptide_mass_lowerbound - 1
  peptide_mass_upperbound_col = peptide_mass_upperbound - 1
  # Search for any nonzero col
  candidate_AA_id = np.flatnonzero(np.any(knapsack_matrix[:, peptide_mass_lowerbound_col:peptide_mass_upperbound_col+1],
                                          axis=1)).tolist()

  return [AA_id + aa_dict_offset for AA_id in candidate_AA_id]


def knapsack_filter(single_aa_candidate, suffix_mass, knapsack_matrix, aa_dict, mass_precision_tolerance=650):
  aa_resolution = 10000
  knapsack_candidate = knapsack_search(knapsack_matrix, suffix_mass, mass_precision_tolerance, aa_resolution, aa_dict)
  # knapsack_candidate2 = knapsack_search(knapsack_matrix, suffix_mass2, mass_precision_tolerance, aa_resolution, aa_dict)
  # knapsack_candidate = list(set(knapsack_candidate1).intersection(set(knapsack_candidate2)))
  aa_candidate = []
  for aa in single_aa_candidate:
    if aa_dict[aa] in knapsack_candidate:
      aa_candidate.append(aa)

  return aa_candidate


def knapsack_mask(tgt, knapsack_matrix, peptide_mass1, peptide_mass2, mass_precision_tolerance, aa_resolution, aa_dict):
  single_aa_candidate1 = knapsack_search(knapsack_matrix, peptide_mass1, mass_precision_tolerance, aa_resolution, aa_dict)
  single_aa_candidate2 = knapsack_search(knapsack_matrix, peptide_mass2, mass_precision_tolerance, aa_resolution, aa_dict)
  single_aa_candidate = list(set(single_aa_candidate1).intersection(set(single_aa_candidate2)))
  predict_mask = torch.zeros_like(tgt)
  predict_mask[single_aa_candidate] = 1
  predict_mask = torch.where(predict_mask.bool(), 0.0, -float('inf'))

  return predict_mask, single_aa_candidate
