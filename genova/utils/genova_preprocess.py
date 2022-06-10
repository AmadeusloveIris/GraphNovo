import sys
import gzip
import torch
import pickle
import numpy as np
import pandas as pd
from glob import glob
from BasicClass import Residual_seq, Ion
from itertools import combinations_with_replacement
from edge_matrix_gen import edge_matrix_generator, typological_sort_floyd_warshall, gen_edge_input

all_edge_mass = []
aalist = Residual_seq.output_aalist()
for num in range(1,7):
    for i in combinations_with_replacement(aalist,num):
        all_edge_mass.append(Residual_seq(i).mass)
all_edge_mass = np.unique(np.array(all_edge_mass))

psm_head = []
for psm_file_name in glob('/data/z37mao/genova_new/*_PSMs.csv'):
    psm_head_temp = pd.read_csv(psm_file_name)
    psm_head_temp['File ID'] = psm_file_name.split('/')[-1][:-9]+':'+psm_head_temp['File ID']
    psm_head.append(psm_head_temp)
psm_head = pd.concat(psm_head)
psm_head = psm_head.set_index('File ID')

with open('candidate_mass','rb') as f:
    candidate_mass = pickle.load(f)
class PeakFeatureGeneration:
    def __init__(self, local_sliding_window, data_acquisition_upper_limit):
        self.local_sliding_window = local_sliding_window
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        
    def __call__(self, product_ions_moverz, product_ions_intensity):
        normalize_moverz = self.normalize_moverzCal(product_ions_moverz)
        relative_intensity = self.relative_intensityCal(product_ions_intensity)
        total_rank = self.total_rankCal(product_ions_intensity)
        total_halfrank = self.total_halfrankCal(product_ions_intensity)
        local_mask = self.local_intensity_mask(product_ions_moverz)
        local_significant = self.local_significantCal(local_mask, product_ions_intensity)
        local_rank = self.local_rankCal(local_mask,product_ions_intensity)
        local_halfrank = self.local_halfrankCal(local_mask,product_ions_intensity)
        local_reletive_intensity = self.local_reletive_intensityCal(local_mask,product_ions_intensity)

        product_ions_feature = np.stack([normalize_moverz,
                                         relative_intensity,
                                         local_significant,
                                         total_rank,
                                         total_halfrank,
                                         local_rank,
                                         local_halfrank,
                                         local_reletive_intensity]).transpose()

        return product_ions_feature
    
    def normalize_moverzCal(self, moverz):
        return np.exp(-moverz/self.data_acquisition_upper_limit)

    def relative_intensityCal(self, intensity):
        return intensity/intensity.max()

    def local_intensity_mask(self, mz):
        right_boundary = np.reshape(mz+self.local_sliding_window,(-1,1))
        left_boundary = np.reshape(mz-self.local_sliding_window,(-1,1))
        mask = np.logical_and(right_boundary>mz,left_boundary<mz)
        return mask

    def local_significantCal(self, mask, intensity): #This feature need to be fixed use signal to ratio to replace intensity.
        #这个feature为了要映射到[1,+infinity)并且不让tan在正无穷和负无穷之间来回横跳，特意在最小intentisy的基础上减了0.5
        #让原始值到不了1
        local_significant=[]
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_significant.append(np.tanh((intensity[i]/local_intensity_list.min()-1)/2))
        return np.array(local_significant)

    def local_rankCal(self, mask, intensity):
        local_rank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_rank.append(np.sum(intensity[i]>local_intensity_list)/len(local_intensity_list))
        return np.array(local_rank)

    def local_halfrankCal(self, mask, intensity):
        local_halfrank = []
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_halfrank.append(np.sum(intensity[i]/2>local_intensity_list)/len(local_intensity_list))
        return np.array(local_halfrank)

    def local_reletive_intensityCal(self, mask, intensity):
        local_reletive_intensity=[]
        for i in range(len(intensity)):
            local_intensity_list = intensity[mask[i]]
            local_reletive_intensity.append(intensity[i]/local_intensity_list.max())
        return np.array(local_reletive_intensity)

    def total_rankCal(self, intensity):
        temp_intensity = intensity.reshape((-1,1))
        return np.sum(temp_intensity>intensity,axis=1)/len(intensity)

    def total_halfrankCal(self, intensity):
        half_intensity = intensity/2
        half_intensity = half_intensity.reshape((-1,1))
        return np.sum(half_intensity>intensity,axis=1)/len(intensity)

class GraphGenerator:
    def __init__(self,
                 candidate_mass,
                 theo_edge_mass,
                 local_sliding_window=50, 
                 data_acquisition_upper_limit=3500,
                 mass_error_da=0.02, 
                 mass_error_ppm=5):
        self.mass_error_da = mass_error_da
        self.mass_error_ppm = mass_error_ppm
        self.theo_edge_mass = theo_edge_mass
        self.candidate_mass = candidate_mass
        self.data_acquisition_upper_limit = data_acquisition_upper_limit
        self.peak_feature_generation = PeakFeatureGeneration(local_sliding_window,data_acquisition_upper_limit)
        self.n_term_ion_list = ['1a','1a-NH3','1a-H2O','1b','1b-NH3','1b-H2O','2a','2a-NH3','2a-H2O','2b','2b-NH3','2b-H2O']
        self.c_term_ion_list = ['1y','1y-NH3','1y-H2O','2y','2y-NH3','2y-H2O']
        
    def __call__(self,product_ions_moverz, product_ions_intensity, precursor_ion_mass, muti_charged):
        peak_feature = self.peak_feature_generation(product_ions_moverz, product_ions_intensity)
        subnode_mass, subnode_feature = self.candidate_subgraph_generator(precursor_ion_mass, product_ions_moverz, peak_feature)
        node_mass = self.graphnode_mass_generator(precursor_ion_mass, product_ions_moverz, muti_charged)
        #assert node_mass.size<=512
        node_feat, node_sourceion = self.graphnode_feature_generator(node_mass, subnode_mass, subnode_feature, precursor_ion_mass)
        subedge_maxnum, edge_type, edge_error = self.edge_generator(node_mass,precursor_ion_mass)
        rel_type, rel_error, dist, rel_pos, rel_coor = self.multihop_rel_generator(subedge_maxnum, edge_type, edge_error)
        graph_label = self.graph_label_generator(seq, node_mass, precursor_ion_mass)
        mask = edge_type>0
        edge_type = edge_type[mask].reshape((-1,1))
        edge_error = edge_error[mask].reshape((-1,1))
        edge_coor = np.array(np.where(mask)).T
        
        node_input = {'node_feat': torch.Tensor(node_feat),
                      'node_sourceion': torch.IntTensor(node_sourceion)}
        
        rel_input = {'rel_type': torch.IntTensor(rel_type),
                     'rel_error': torch.Tensor(rel_error),
                     'dist': torch.IntTensor(dist),
                     'rel_pos': torch.IntTensor(rel_pos),
                     'rel_coor': torch.LongTensor(rel_coor)}
        
        edge_input = {'edge_type': torch.IntTensor(edge_type),
                      'edge_error': torch.Tensor(edge_error),
                      'edge_coor': torch.LongTensor(edge_coor)}
        
        graph_label = torch.IntTensor(graph_label)
        
        return node_mass, node_input, rel_input, edge_input, graph_label
    
    def candidate_subgraph_generator(self, precursor_ion_mass, product_ions_moverz, product_ions_feature):
        candidate_subgraphnode_moverz = []
        candidate_subgraphnode_moverz += [Ion.peak2sequencemz(product_ions_moverz,ion) for ion in self.n_term_ion_list]
        candidate_subgraphnode_moverz += [precursor_ion_mass-Ion.peak2sequencemz(product_ions_moverz,ion) for ion in self.c_term_ion_list]
        candidate_subgraphnode_moverz = np.concatenate(candidate_subgraphnode_moverz)
        candidate_subgraphnode_feature = []
        for i in range(2,len(self.n_term_ion_list)+len(self.c_term_ion_list)+2):
            candidate_subgraphnode_source = i*np.ones([product_ions_moverz.size, 1])
            candidate_subgraphnode_feature.append(np.concatenate((product_ions_feature,candidate_subgraphnode_source),axis=1))
        candidate_subgraphnode_feature = np.concatenate(candidate_subgraphnode_feature)
        
        candidate_subgraphnode_moverz = np.insert(candidate_subgraphnode_moverz,
                                                  [0,candidate_subgraphnode_moverz.size],
                                                  [0,precursor_ion_mass])
        sorted_index = np.argsort(candidate_subgraphnode_moverz)
        candidate_subgraphnode_moverz = candidate_subgraphnode_moverz[sorted_index]
        
        candidate_subgraphnode_feature = np.concatenate([np.array([1]*9).reshape(1,-1),
                                                         candidate_subgraphnode_feature,
                                                         np.array([1]*9).reshape(1,-1)],axis=0)
        candidate_subgraphnode_feature = candidate_subgraphnode_feature[sorted_index]
        return candidate_subgraphnode_moverz, candidate_subgraphnode_feature

    def record_filter(self, mass_list, precursor_ion_mass=None):
        if precursor_ion_mass:
            mass_threshold = self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        else:
            mass_threshold = self.mass_error_da
        
        mask = self.candidate_mass.searchsorted(mass_list-mass_threshold)!=self.candidate_mass.searchsorted(mass_list+mass_threshold)
        mask = np.logical_or(mask,mass_list>=self.candidate_mass.max())
        return mass_list[mask], mask

    def graphnode_mass_generator(self, precursor_ion_mass, product_ions_moverz, muti_charged):
        #1a ion
        node_1a_mass_nterm, _ = self.record_filter(Ion.peak2sequencemz(product_ions_moverz[product_ions_moverz<250],'1a'))
        _, mask = self.record_filter(precursor_ion_mass-node_1a_mass_nterm,precursor_ion_mass)
        node_1a_mass_nterm = node_1a_mass_nterm[mask]
        #1b ion
        node_1b_mass_nterm, _ = self.record_filter(Ion.peak2sequencemz(product_ions_moverz,'1b'))
        _, mask = self.record_filter(precursor_ion_mass-node_1b_mass_nterm,precursor_ion_mass)
        node_1b_mass_nterm = node_1b_mass_nterm[mask]
        #1y ion
        node_1y_mass_cterm, _ = self.record_filter(Ion.peak2sequencemz(product_ions_moverz,'1y'))
        node_1y_mass_nterm, _ = self.record_filter(precursor_ion_mass-node_1y_mass_cterm,precursor_ion_mass)
        #2y ion
        node_2y_mass_cterm, _ = self.record_filter(Ion.peak2sequencemz(product_ions_moverz[product_ions_moverz>400],'2y'))
        node_2y_mass_nterm, _ = self.record_filter(precursor_ion_mass-node_2y_mass_cterm,precursor_ion_mass)
        if muti_charged:
            graphnode_mass = np.concatenate([node_1a_mass_nterm,node_1b_mass_nterm,node_1y_mass_nterm,node_2y_mass_nterm])
        else:
            graphnode_mass = np.concatenate([node_1a_mass_nterm,node_1b_mass_nterm,node_1y_mass_nterm])
        graphnode_mass = np.unique(graphnode_mass)
        graphnode_mass = np.insert(graphnode_mass,
                                   [0,graphnode_mass.size],
                                   [0,precursor_ion_mass])
        return graphnode_mass
    
    def graphnode_feature_generator(self, graphnode_mass, subnode_mass, subnode_feature, precursor_ion_mass):
        mass_threshold = 2*self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        lower_bounds = subnode_mass.searchsorted(graphnode_mass - mass_threshold)
        higher_bounds = subnode_mass.searchsorted(graphnode_mass + mass_threshold)
        subnode_maxnum = (higher_bounds-lower_bounds).max()
        node_feature = []
        for i,(lower_bound,higher_bound) in enumerate(zip(lower_bounds,higher_bounds)):
            mass_merge_error = np.abs(graphnode_mass[i] - subnode_mass[np.arange(lower_bound,higher_bound)])
            mass_merge_index = np.argsort(mass_merge_error)
            mass_merge_error = np.exp(-np.abs(mass_merge_error)/mass_threshold)
            mass_merge_feat = np.concatenate([np.exp(-graphnode_mass[i]*np.ones((higher_bound-lower_bound,1))/self.data_acquisition_upper_limit),
                                              subnode_feature[np.arange(lower_bound,higher_bound)],
                                              mass_merge_error.reshape(-1,1)],axis=1)
            mass_merge_feat = mass_merge_feat[mass_merge_index]
            mass_merge_feat = np.pad(mass_merge_feat,((0,subnode_maxnum-(higher_bound-lower_bound)),(0,0)))
            node_feature.append(mass_merge_feat)
        node_feature = np.stack(node_feature)
        node_feature[0,1:,:]=0
        node_sourceion = node_feature[:,:,-2]
        node_feat = np.delete(node_feature,-2,axis=2)
        return node_feat, node_sourceion
    
    def edge_generator(self, graphnode_moverz, precursor_ion_mass):
        n = graphnode_moverz.size
        mass_threshold = 2*self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        mass_difference = np.zeros((n,n),dtype=np.float64)
        for x in range(graphnode_moverz.size-1):
            mass_difference[x,x+1:] = graphnode_moverz[x+1:] - graphnode_moverz[x]
        start_edge_type = self.theo_edge_mass.searchsorted(mass_difference-mass_threshold)
        end_edge_type = self.theo_edge_mass.searchsorted(mass_difference+mass_threshold)

        #######
        #边信息构建
        subedge_maxnum = np.max(end_edge_type-start_edge_type)
        edge_type, edge_error = edge_matrix_generator(n,
                                                      mass_threshold,
                                                      subedge_maxnum,
                                                      self.theo_edge_mass,
                                                      mass_difference,
                                                      start_edge_type,
                                                      end_edge_type)
        
        return subedge_maxnum, edge_type, edge_error
    
    def multihop_rel_generator(self, subedge_maxnum, edge_type, edge_error):
        n = edge_type.shape[0]
        adjacency_matrix = np.any(edge_type,axis=-1).astype(int)
        
        dist, predecessors = typological_sort_floyd_warshall(n, adjacency_matrix)

        max_dist = dist.max()
        rel_type, rel_error = gen_edge_input(n,
                                             max_dist,
                                             subedge_maxnum,
                                             predecessors,
                                             edge_type,
                                             edge_error,
                                             adjacency_matrix)

        mask = rel_type>0
        rel_type = rel_type[mask].reshape((-1,1))
        rel_error = rel_error[mask].reshape((-1,1))
        rel_coor = np.array(np.where(mask)).T
        rel_pos = rel_coor[-2]
        
        return rel_type, rel_error, dist, rel_pos, rel_coor
    
    def graph_label_generator(self, seq, node_mass, precursor_ion_mass):
        theo_node_mass = np.insert(Residual_seq(seq).step_mass,0,0)
        mass_threshold = self.mass_error_da+self.mass_error_ppm*precursor_ion_mass*1e-6
        start_index = node_mass.searchsorted(theo_node_mass-mass_threshold)
        end_index = node_mass.searchsorted(theo_node_mass+mass_threshold)
        graph_label = np.zeros((node_mass.size,theo_node_mass.size))
        for i, (lower_bound, higher_bound) in enumerate(zip(start_index, end_index)):
            graph_label[:,i][lower_bound:higher_bound] = 1
        return graph_label

graph_gen = GraphGenerator(candidate_mass,all_edge_mass)

if __name__=='__main__':
    worker, start_i, end_i = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    psm_head = psm_head.iloc[start_i:end_i]
    with open('/home/z37mao/genova_data/{}.csv'.format(worker), 'w') as index_writer:
        index_writer.write('Spec Index,Node Number,Relation Num,Edge Num,MSGP File Name,MSGP Datablock Pointer,MSGP Datablock Length\n')
        for i, (spec_index, (experiment_name, seq, precursor_charge, precursor_moverz, pointer, data_len)) in enumerate(psm_head[['MGFS Experiment Name','Annotated Sequence','Charge','m/z [Da]','MGFS_Datablock_Pointer','MGFS_Datablock_Length']].iterrows()):
            file_num = i//4000
            if i%4000==0:
                try: writer.close()
                except: pass 
                writer = open('/home/z37mao/genova_data/{}_{}.msgp'.format(worker, file_num),'wb')
            with open('/data/z37mao/genova_new/{}.mgfs'.format(experiment_name),'rb') as f:
                f.seek(pointer)
                seq = seq.replace('L','I')
                product_ion_info = pickle.loads(f.read(data_len))
                precursor_ion_mass = Ion.precursorion2mass(precursor_moverz, precursor_charge)
                product_ions_moverz, product_ions_intensity = product_ion_info['product_ions_moverz'], product_ion_info['product_ions_intensity']
                node_mass, node_input, rel_input, edge_input, graph_label = graph_gen(product_ions_moverz, product_ions_intensity, precursor_ion_mass, precursor_charge>2)
                record = {'node_mass':node_mass,
                          'node_input':node_input, 
                          'rel_input':rel_input, 
                          'edge_input':edge_input, 
                          'graph_label':graph_label}
                compressed_data = gzip.compress(pickle.dumps(record))
                index_writer.write('{},{},{},{},{},{},{}\n'.format(spec_index,node_mass.size,len(rel_input['rel_type']),len(edge_input['edge_type']),"{}_{}.msgp".format(worker, file_num),writer.tell(),len(compressed_data)))
                writer.write(compressed_data)
                