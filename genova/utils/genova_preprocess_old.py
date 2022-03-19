import zmq
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from BasicClass import Residual_seq, Ion
from itertools import combinations_with_replacement, product
from edge_matrix_gen import edge_matrix_generator, typological_sort_floyd_warshall, gen_edge_input 

def label_ion_verification(theo_moverz_list, product_ion_moverz_list):
    #这个函数是为了检测二级谱图所对应的序列有没有对应离子。
    real_moverz = np.append(product_ion_moverz_list, np.inf)
    index = np.searchsorted(real_moverz, theo_moverz_list)
    return np.logical_or(np.abs(real_moverz[index]-theo_moverz_list)<0.02,np.abs(real_moverz[index-1]-theo_moverz_list)<0.02)

def spec_verification(seq,product_ions_moverz,muti_charged):
    #验证二级质谱是否符合制图条件
    nterm = Residual_seq(seq).step_mass[:-1]
    cterm = Residual_seq(seq).mass-Residual_seq(seq).step_mass[:-1]

    a1ion_theomoverz = Ion.sequencemz2ion(nterm,'1a')
    b1ion_theomoverz = Ion.sequencemz2ion(nterm,'1b')
    y1ion_theomoverz = Ion.sequencemz2ion(cterm,'1y')

    a1_existed = label_ion_verification(a1ion_theomoverz,product_ions_moverz[product_ions_moverz<200])
    b1_existed = label_ion_verification(b1ion_theomoverz,product_ions_moverz)
    y1_existed = label_ion_verification(y1ion_theomoverz,product_ions_moverz)

    if muti_charged:
        y2ion_theomoverz = Ion.sequencemz2ion(cterm,'2y')
        y2_existed = label_ion_verification(y2ion_theomoverz,product_ions_moverz[product_ions_moverz>400])
        seq_ion_existed = a1_existed+b1_existed+y1_existed+y2_existed
    else:
        seq_ion_existed = a1_existed+b1_existed+y1_existed

    non_existed_pos = np.where(np.logical_not(seq_ion_existed))[0]
    return np.any((non_existed_pos[1:]-non_existed_pos[:-1])==1)

def graph_verification(seq,graphnode_moverz):
    seq_node_existed = label_ion_verification(Residual_seq(seq).step_mass[:-1],graphnode_moverz)
    non_existed_pos = np.where(np.logical_not(seq_node_existed))[0]
    return np.any((non_existed_pos[1:]-non_existed_pos[:-1])==1)

def record_filter(theo_moverz_list, moverz_list, type_flag=None):
    #用预先生成的a，b，y ion list对离子进行过滤，将一部分噪音离子过滤掉
    index = np.searchsorted(theo_moverz_list, moverz_list)
    mask = np.logical_or(np.abs(theo_moverz_list[index]-moverz_list)<0.02,np.abs(theo_moverz_list[index-1]-moverz_list)<0.02)
    if type_flag=='ion':
        mask = np.logical_or(mask,moverz_list>=400)
        return moverz_list[mask]
    elif type_flag=='graph':
        mask = np.logical_or(mask,moverz_list>=1000)
        return moverz_list[mask]

def graphnode_moverz_generator(theo_graphnode_list, precursor_ion_moverz, precursor_ion_charge, product_ions_moverz, muti_charged):
    seqnode_1a_index = np.where(product_ions_moverz<200)[0]
    seqnode_1a_nterm = Ion.peak2sequencemz(product_ions_moverz[seqnode_1a_index],'1a')
    seqnode_1b_nterm = Ion.peak2sequencemz(product_ions_moverz,'1b')
    seqnode_1y_cterm = Ion.peak2sequencemz(product_ions_moverz,'1y')
    seqnode_1y_nterm = Ion.precursorion2mass(precursor_ion_moverz,precursor_ion_charge) - seqnode_1y_cterm

    if muti_charged:
        seqnode_2y_index = np.where(product_ions_moverz>400)[0]
        seqnode_2y_cterm = Ion.peak2sequencemz(product_ions_moverz[seqnode_2y_index],'2y')
        seqnode_2y_nterm = Ion.precursorion2mass(precursor_ion_moverz,precursor_ion_charge) - seqnode_2y_cterm
        graphnode_moverz = np.concatenate([seqnode_1a_nterm,seqnode_1b_nterm,seqnode_1y_nterm,seqnode_2y_nterm])
    else:
        graphnode_moverz = np.concatenate([seqnode_1a_nterm,seqnode_1b_nterm,seqnode_1y_nterm])

    graphnode_moverz = record_filter(theo_graphnode_list, graphnode_moverz, type_flag='graph')
    graphnode_moverz = graphnode_moverz[graphnode_moverz<Ion.precursorion2mass(precursor_ion_moverz,precursor_ion_charge)]
    graphnode_moverz.sort()
    return graphnode_moverz

def candidate_subgraph_generator(precursor_ion_moverz, precursor_ion_charge, product_ions_moverz, product_ions_feature):
    candidate_subgraphnode_moverz = []

    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1a'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1a-NH3'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1a-H2O'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1b'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1b-NH3'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'1b-H2O'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'1y'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'1y-NH3'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'1y-H2O'))

    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2a'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2a-NH3'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2a-H2O'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2b'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2b-NH3'))
    candidate_subgraphnode_moverz.append(Ion.peak2sequencemz(product_ions_moverz,'2b-H2O'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'2y'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'2y-NH3'))
    candidate_subgraphnode_moverz.append(Ion.precursorion2mass(precursor_ion_moverz, precursor_ion_charge)-Ion.peak2sequencemz(product_ions_moverz,'2y-H2O'))

    candidate_subgraphnode_moverz = np.concatenate(candidate_subgraphnode_moverz)

    candidate_subgraphnode_feature = []
    for i in range(1,19):
        candidate_subgraphnode_source = np.zeros([product_ions_moverz.size, 1])
        candidate_subgraphnode_source[:,0] = i
        candidate_subgraphnode_feature.append(np.concatenate((product_ions_feature,candidate_subgraphnode_source),axis=1))
    candidate_subgraphnode_feature = np.concatenate(candidate_subgraphnode_feature)
    sorted_index = np.argsort(candidate_subgraphnode_moverz)

    return candidate_subgraphnode_moverz[sorted_index], candidate_subgraphnode_feature[sorted_index]

def graphnode_generator(theo_ion_moverz, theo_graphnode_moverz, precursor_ion_moverz, precursor_ion_charge, product_ions_moverz, product_ions_feature, muti_charged):
    filted_product_ions_moverz = record_filter(theo_ion_moverz, product_ions_moverz, 'ion')
    graphnode_moverz = graphnode_moverz_generator(theo_graphnode_moverz, precursor_ion_moverz, precursor_ion_charge, filted_product_ions_moverz, muti_charged)
    candidate_subgraph_node_moverz, candidate_subgraph_node_feature = candidate_subgraph_generator(precursor_ion_moverz, precursor_ion_charge, product_ions_moverz, product_ions_feature)

    start_indexes = np.searchsorted(candidate_subgraph_node_moverz,graphnode_moverz-0.04)
    end_indexes = np.searchsorted(candidate_subgraph_node_moverz,graphnode_moverz+0.04)
    max_subnodegraph_num = np.max(end_indexes-start_indexes)
    graphnode_feature = []

    for i in range(graphnode_moverz.size):
        subgraph_node_feature_buffer = np.zeros((max_subnodegraph_num,10))
        subgraph_length = end_indexes[i] - start_indexes[i]
        subgraph_node_feature_buffer[:subgraph_length,:9] = candidate_subgraph_node_feature[start_indexes[i]:end_indexes[i]]
        subgraph_node_feature_buffer[:subgraph_length, -1] = np.exp(-np.abs(graphnode_moverz[i]-candidate_subgraph_node_moverz[start_indexes[i]:end_indexes[i]])/0.04)
        graphnode_feature.append(subgraph_node_feature_buffer)

    #add start and end vertices
    start_graphnode_feature = np.zeros([max_subnodegraph_num,10])
    start_graphnode_feature[0] = np.array([1]+[1]*7+[19]+[1])
    end_graphnode_feature = np.zeros([max_subnodegraph_num,10])
    end_graphnode_feature[0] = np.array([np.exp(-Ion.precursorion2mass(precursor_ion_moverz,precursor_ion_charge)/3500)]+[1]*7+[19]+[1])
    graphnode_feature = [start_graphnode_feature] + graphnode_feature + [end_graphnode_feature]
    graphnode_feature = np.stack(graphnode_feature)

    graphnode_moverz = np.insert(graphnode_moverz,0,0)
    graphnode_moverz = np.append(graphnode_moverz, Ion.precursorion2mass(precursor_ion_moverz,precursor_ion_charge))
    return graphnode_moverz, graphnode_feature

##All candidate feature

def normalize_moverzCal(moverz, data_acquisition_upper_limit):
    return np.exp(-moverz/data_acquisition_upper_limit)

def relative_intensityCal(intensity):
    return intensity/intensity.max()

def local_intensity_mask(mz):
    right_boundary = np.reshape(mz+50,(-1,1))
    left_boundary = np.reshape(mz-50,(-1,1))
    mask = np.logical_and(right_boundary>mz,left_boundary<mz)
    return mask

def local_significantCal(mask, intensity): #This feature need to be fixed use signal to ratio to replace intensity.
    #这个feature为了要映射到[1,+infinity)并且不让tan在正无穷和负无穷之间来回横跳，特意在最小intentisy的基础上减了0.5
    #让原始值到不了1
    local_significant=[]
    for i in range(len(intensity)):
        local_intensity_list = intensity[mask[i]]
        local_significant.append(np.tanh((intensity[i]/local_intensity_list.min()-1)/2))
    return np.array(local_significant)

def local_rankCal(mask,intensity):
    local_rank = []
    for i in range(len(intensity)):
        local_intensity_list = intensity[mask[i]]
        local_rank.append(np.sum(intensity[i]>local_intensity_list)/len(local_intensity_list))
    return np.array(local_rank)

def local_halfrankCal(mask,intensity):
    local_halfrank = []
    for i in range(len(intensity)):
        local_intensity_list = intensity[mask[i]]
        local_halfrank.append(np.sum(intensity[i]/2>local_intensity_list)/len(local_intensity_list))
    return np.array(local_halfrank)

def local_reletive_intensityCal(mask,intensity):
    local_reletive_intensity=[]
    for i in range(len(intensity)):
        local_intensity_list = intensity[mask[i]]
        local_reletive_intensity.append(intensity[i]/local_intensity_list.max())
    return np.array(local_reletive_intensity)

def total_rankCal(intensity):
    temp_intensity = intensity.reshape((-1,1))
    return np.sum(temp_intensity>intensity,axis=1)/len(intensity)

def total_halfrankCal(intensity):
    half_intensity = intensity/2
    half_intensity = half_intensity.reshape((-1,1))
    return np.sum(half_intensity>intensity,axis=1)/len(intensity)

def feature_genrator(product_ions_moverz, product_ions_intensity, data_acquisition_upper_limit):
    normalize_moverz = normalize_moverzCal(product_ions_moverz, data_acquisition_upper_limit)
    relative_intensity = relative_intensityCal(product_ions_intensity)
    total_rank = total_rankCal(product_ions_intensity)
    total_halfrank = total_halfrankCal(product_ions_intensity)
    local_mask = local_intensity_mask(product_ions_moverz)
    local_significant = local_significantCal(local_mask, product_ions_intensity)
    local_rank = local_rankCal(local_mask,product_ions_intensity)
    local_halfrank = local_halfrankCal(local_mask,product_ions_intensity)
    local_reletive_intensity = local_reletive_intensityCal(local_mask,product_ions_intensity)

    product_ions_feature = np.stack([normalize_moverz,
                                     relative_intensity,
                                     local_significant,
                                     total_rank,
                                     total_halfrank,
                                     local_rank,
                                     local_halfrank,
                                     local_reletive_intensity]).transpose()

    return product_ions_feature

def graph_edge_filter(adjacent_matrix):
    num_node = adjacent_matrix.shape[0]
    keep_mask = np.ones_like(adjacent_matrix,dtype=bool)
    for y in range(1,num_node-1):
        if not np.any(np.logical_and(keep_mask[:,y],adjacent_matrix[:,y])):
            keep_mask[:, y] = False
            keep_mask[y, :] = False
    for x in range(num_node-2,0,-1):
        if not np.any(np.logical_and(keep_mask[x,:],adjacent_matrix[x,:])):
            keep_mask[x, :] = False
            keep_mask[:, x] = False
    return keep_mask

def graphnode_equivalence_filter(start_edge_type, end_edge_type, graphnode_moverz):
    num_node = graphnode_moverz.size

    equal_node = {}
    registered_node = {i:0 for i in range(1,num_node-1)}

    for start_index in range(1,num_node-2):
        if registered_node[start_index]!=0: continue
        for search_index in range(start_index+1,num_node-1):
            if graphnode_moverz[search_index]-graphnode_moverz[start_index]>0.04: break
            if np.all(np.equal(end_edge_type[:,start_index][:start_index],end_edge_type[:,search_index][:start_index])):
                if np.all(np.equal(start_edge_type[:,start_index][:start_index],start_edge_type[:,search_index][:start_index])):
                    if np.all(np.equal(end_edge_type[start_index][search_index+1:],end_edge_type[search_index][search_index+1:])):
                        if np.all(np.equal(start_edge_type[start_index][search_index+1:],start_edge_type[search_index][search_index+1:])):
                            if np.all(start_edge_type[start_index][start_index+1:search_index+1]==end_edge_type[start_index][start_index+1:search_index+1]):
                                if np.all(start_edge_type[:,search_index][start_index:search_index]==end_edge_type[:,search_index][start_index:search_index]):
                                    try:
                                        equal_node[start_index].append(search_index)
                                        registered_node[search_index] = start_index
                                    except KeyError:
                                        equal_node[start_index] = [start_index, search_index]
                                        registered_node[search_index] = start_index

    delete_index = []
    for equivalent_graphnode_group in equal_node.values():
        equivalent_graphnode_group.pop(math.ceil(len(equivalent_graphnode_group)/2)-1)
        delete_index+=equivalent_graphnode_group
    equivalent_graphnode_mask = np.ones((num_node, num_node),dtype=bool)
    equivalent_graphnode_mask[:,delete_index]=False
    equivalent_graphnode_mask[delete_index,:]=False
    return equivalent_graphnode_mask

def edge_generator(theo_edge_mass, graphnode_moverz, graphnode_feature):
    mass_difference = np.zeros((graphnode_moverz.size,graphnode_moverz.size),dtype=np.float64)
    for x in range(graphnode_moverz.size-1):
        mass_difference[x,x+1:] = graphnode_moverz[x+1:] - graphnode_moverz[x]
    start_edge_type = theo_edge_mass.searchsorted(mass_difference-0.04)
    end_edge_type = theo_edge_mass.searchsorted(mass_difference+0.04)

    #######
    #点连通性筛选
    mask = graph_edge_filter((end_edge_type-start_edge_type)>0)
    start_edge_type = start_edge_type[mask].reshape((mask[0].sum(),mask[0].sum()))
    end_edge_type = end_edge_type[mask].reshape((mask[0].sum(),mask[0].sum()))
    mass_difference = mass_difference[mask].reshape((mask[0].sum(),mask[0].sum()))
    graphnode_moverz, graphnode_feature = graphnode_moverz[mask[0]], graphnode_feature[mask[0]]

    #######
    #点等价性筛选
    equivalent_graphnode_mask = graphnode_equivalence_filter(start_edge_type, end_edge_type, graphnode_moverz)
    start_edge_type = start_edge_type[equivalent_graphnode_mask].reshape((equivalent_graphnode_mask[0].sum(),
                                                                          equivalent_graphnode_mask[0].sum()))
    end_edge_type = end_edge_type[equivalent_graphnode_mask].reshape((equivalent_graphnode_mask[0].sum(),
                                                                      equivalent_graphnode_mask[0].sum()))
    mass_difference = mass_difference[equivalent_graphnode_mask].reshape((equivalent_graphnode_mask[0].sum(),
                                                                          equivalent_graphnode_mask[0].sum()))
    graphnode_moverz, graphnode_feature = graphnode_moverz[equivalent_graphnode_mask[0]], graphnode_feature[equivalent_graphnode_mask[0]]
    #######
    #边信息构建
    n = graphnode_moverz.size
    subedge_maxnum = np.max(end_edge_type-start_edge_type)
    edge_matrix, edge_rdifferent_matrix = edge_matrix_generator(n,
                                                                subedge_maxnum,
                                                                theo_edge_mass,
                                                                mass_difference,
                                                                start_edge_type,
                                                                end_edge_type)

    adjacency_matrix = ((end_edge_type-start_edge_type)>0).astype(int)
    dist_matrix, predecessors = typological_sort_floyd_warshall(n, adjacency_matrix)

    max_dist = dist_matrix.max()
    path_matrix, path_rdifferent_matrix = gen_edge_input(n,
                                                         max_dist,
                                                         subedge_maxnum,
                                                         predecessors,
                                                         edge_matrix,
                                                         edge_rdifferent_matrix,
                                                         adjacency_matrix)

    path_mask = dist_matrix>0+np.identity(n,dtype=bool)

    return dist_matrix, edge_matrix, edge_rdifferent_matrix, path_mask, path_matrix, path_rdifferent_matrix, adjacency_matrix, graphnode_moverz, graphnode_feature

def path_label_generator(seq, graphnode_moverz):
    start_index = graphnode_moverz.searchsorted(Residual_seq(seq).step_mass[:-1]-0.02)
    end_index = graphnode_moverz.searchsorted(Residual_seq(seq).step_mass[:-1]+0.02)
    if np.prod((end_index-start_index)[(end_index-start_index)>1])<200:
        temp_node_position = [[0]]
        for i in range(len(seq)-1):
            temp_node_position.append([node for node in range(start_index[i],end_index[i])])
        temp_node_position.append([graphnode_moverz.size-1])
        possible_node_path = []
        for i in product(*temp_node_position):
            possible_node_path.append(np.array(i))
        possible_node_path = np.array(possible_node_path)
        return possible_node_path
    else:
        return np.array([])

def seq_label_generator(aa_dict, seq):
    seq_label = np.array([aa_dict[aa] for aa in seq])
    return seq_label

if __name__=='__main__':
    context = zmq.Context()
    receive = context.socket(zmq.PULL)
    receive.connect('tcp://127.0.0.1:5557')


    #experiment_name = 'Plasma'
    data_acquisition_upper_limit = 3500
    #psm_head = pd.read_csv('./data/{}/{}_PSMs.csv'.format(experiment_name,experiment_name))
    theo_ion_moverz = np.load('possible_ion_moverz')
    theo_graphnode_moverz = np.load('possible_graphnode_moverz')
    
    theo_edge_mass = []
    aalist = Residual_seq.output_aalist()
    for num in range(1,3):
        for i in combinations_with_replacement(aalist,num):
            theo_edge_mass.append(Residual_seq(i).mass)
    theo_edge_mass = np.array(sorted(set([float(-100)]+theo_edge_mass+[float("inf")])))
    
    aa_dict = {aa:i for i, aa in enumerate(Residual_seq._Residual_seq__aa_residual_composition,start=1)}

    #for _, (file_id, scan) in tqdm(psm_head[['File ID','Scan']].iterrows()):
    while True:
        data = receive.recv_json()
        experiment_name, file_id, scan = data['experiment_name'], data['file_id'], data['scan']
        with open('./data/{}/{}/{}.ms2'.format(experiment_name, file_id, scan),'rb') as f:
            #data load
            ms2_spectrum = pickle.load(f)
            seq = ms2_spectrum['seq']
            seq = seq.replace('L','I')
            precursor_ion_moverz,precursor_ion_charge = ms2_spectrum['precursor_ion_moverz'],ms2_spectrum['precursor_ion_charge']
            product_ions_moverz, product_ions_intensity = np.array(ms2_spectrum['product_ions_moverz']), np.array(ms2_spectrum['product_ions_intensity'])

            #graph vertices generation
            muti_charged = precursor_ion_charge>2
            if spec_verification(seq,product_ions_moverz,muti_charged): continue
            product_ions_feature = feature_genrator(product_ions_moverz, product_ions_intensity, data_acquisition_upper_limit)
            #product_ions_moverz, product_ions_feature = record_filter(theo_ion_moverz, product_ions_moverz, product_ions_feature, 'ion')
            graphnode_moverz, graphnode_feature = graphnode_generator(theo_ion_moverz, theo_graphnode_moverz, precursor_ion_moverz, precursor_ion_charge, product_ions_moverz, product_ions_feature, muti_charged)
            if graph_verification(seq, graphnode_moverz): continue
            if graphnode_moverz.size>1024: continue
            dist_matrix, edge_matrix, edge_rdifferent_matrix, path_mask, path_matrix, path_rdifferent_matrix, adjacency_matrix, graphnode_moverz, graphnode_feature = edge_generator(theo_edge_mass, graphnode_moverz, graphnode_feature)
            path_label = path_label_generator(seq, graphnode_moverz)
            seq_label = seq_label_generator(aa_dict, seq)
            np.savez_compressed('/home/z37mao/genova/pretrain_data/{}/{}/{}'.format(experiment_name, file_id, scan),
                                graphnode_moverz=graphnode_moverz,
                                graphnode_feature=graphnode_feature,
                                edge_matrix=edge_matrix,
                                edge_rdifferent_matrix=edge_rdifferent_matrix,
                                path_mask=path_mask,
                                path_matrix=path_matrix,
                                path_rdifferent_matrix=path_rdifferent_matrix,
                                adjacency_matrix=adjacency_matrix,
                                path_label=path_label,
                                seq_label=seq_label)
