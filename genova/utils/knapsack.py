import pickle
from sys import argv
from collections import OrderedDict

from sklearn.preprocessing import OrdinalEncoder
from BasicClass import Residual_seq

aa_list = Residual_seq.output_aalist()

def knapsack_composition_builder(max_mass: float):
    knapsack_composition = {}
    for aa in aa_list:
        result = {}
        sorted_composition_list = sorted(knapsack_composition.keys())
        for i in range(1,int(max_mass/Residual_seq(aa).mass)+1):
            aa_composition = Residual_seq(i*aa).composition
            if aa_composition in result: result[aa_composition] = set.union(set(aa),result[aa_composition])
            else: result[aa_composition] = set(aa)
            for candidate_composition in sorted_composition_list:
                if (candidate_composition + aa_composition).mass > max_mass: break
                next_composition = candidate_composition + aa_composition
                if next_composition in result: result[next_composition] = set.union(result[next_composition], set(aa).union(knapsack_composition[candidate_composition]))
                else: result[next_composition] = set.union(set(aa),knapsack_composition[candidate_composition])
        
        for candidate_composition in result:
            if candidate_composition in knapsack_composition: 
                knapsack_composition[candidate_composition] = set.union(knapsack_composition[candidate_composition],result[candidate_composition])
            else:
                knapsack_composition[candidate_composition] = result[candidate_composition]
    return knapsack_composition

if __name__=='__main__':
    knapsack_composition = knapsack_composition_builder(argv[1])
    pickle.dump(knapsack_composition,open(argv[2],'wb'))
    
    knapsack_mass = OrderedDict()
    temp = {composition.mass:composition for composition in knapsack_composition.keys()}
    for candidate_mass in sorted(temp.keys()):
        knapsack_mass[candidate_mass] = knapsack_composition[temp[candidate_mass]]
    pickle.dump(knapsack_mass,open(argv[3],'wb'))