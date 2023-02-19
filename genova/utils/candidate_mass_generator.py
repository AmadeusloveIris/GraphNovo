import pickle
import numpy as np
from BasicClass import Composition, Residual_seq, Ion
aalist = Residual_seq.output_aalist()

def all_possible_seqmz(max_mass, aalist_startindex):
    i = 0
    result = []
    if len(aalist)-1 == aalist_startindex:
        while True:
            if i == 0: mass = 0
            else: mass = Residual_seq([aalist[aalist_startindex]] * i).mass
            mass_remain = max_mass - mass
            if mass_remain < 0: return result
            result += [mass]
            i+=1

    while True:
        if i==0: mass=0
        else: mass = Residual_seq([aalist[aalist_startindex]] * i).mass
        mass_remain = max_mass - mass
        if mass_remain < 0: return result
        for r in all_possible_seqmz(mass_remain, aalist_startindex + 1):
            result += [mass+r]
        i+=1

result_seqmz = all_possible_seqmz(1000,0)
result_seqmz = np.sort(np.unique(np.round(np.array(result_seqmz),8)))[1:]
pickle.dump(result_seqmz, open('candidate_mass','wb'))
