import math
import torch
import numpy as np
from torch.nn.functional import pad
from BasicClass import Residual_seq, Ion
from torch.utils.data import Dataset

class ValidDataset(Dataset):
    def __init__(self, aa_dict, data, psm_head, tokenize_aa_dict, max_len=32, hidden_size=768, max_point=256, nhead=12):
        super().__init__()
        self.aa_dict = aa_dict
        self.data = data
        self.psm_head = psm_head
        self.tokenize_aa_dict = tokenize_aa_dict 
        self.single_aa_mask_list = ['N','Q','K','m','F','R','W']
        self.max_len = max_len*3
        self.psm_head = self.psm_head[self.psm_head['Annotated Sequence'].str.split(' ').str.join('').str.len()<=self.max_len]
        self.hidden_size = hidden_size
        self.max_point = max_point
        self.nhead = nhead
        self.aa_candidate_mass = np.array([Residual_seq(aa).mass for aa in Residual_seq.output_aalist()])
        
    def __getitem__(self, idx):
        seq,moverz,charge,experiment,file_id,scan,_ = self.psm_head.iloc[idx]
        label_seq = seq
        inference_seq = seq
        ion_moverz_encoder, ion_moverz_decoder, intensity, ion_mask_encoder, ion_mask_decoder = self.spectra_processor(experiment,file_id,scan)
        label_mask, tgt_mask = self.decoder_mask(label_seq, inference_seq)
        label, tgt = self.tonkenize(label_seq, inference_seq)
        pepblock_index, aa_index_in_block = self.token_position(inference_seq)
        peptide_mass_embedding = self.peptide_mass_embedding(inference_seq, moverz, charge)
        return ({'src':{'src':intensity,
                        'charge':charge},
                 'tgt':{'tgt':tgt,
                        'pepblock_index':pepblock_index,
                        'aa_index_in_block':aa_index_in_block},
                 'tgt_mask': tgt_mask,
                'encoder_input':{'peaks_moverz':ion_moverz_encoder,
                                 'src_key_padding_mask':ion_mask_encoder},
                'crossattn':{'mem_key_padding_mask':ion_mask_decoder,
                             'pep_mass':peptide_mass_embedding,
                             'peaks_moverz':ion_moverz_decoder}},
                {'label':label,'label_mask':label_mask})
        
    def __len__(self):
        return len(self.psm_head)

    def spectra_processor(self,experiment,file_id,scan):
        ion_moverz, intensity = self.data[':'.join([experiment,file_id,str(scan)])]
        spectra_mask_encoder = torch.zeros([1,self.max_point],dtype=bool)
        spectra_mask_encoder[...,len(ion_moverz):] = True
        spectra_mask_decoder = spectra_mask_encoder.unsqueeze(0)
        ion_moverz = np.pad(ion_moverz,(1,self.max_point-len(ion_moverz)-1))
        intensity = np.pad(intensity,(1,self.max_point-len(intensity)-1))
        intensity = intensity/intensity.max()
        ion_moverz_encoder = self.sinusoidal_position_embedding(ion_moverz, 128, 1e4, 1e-5)
        ion_moverz_decoder_sin, ion_moverz_decoder_cos = self.sinusoidal_position_embedding(ion_moverz, 64, 1e4, 1e-5)
        ion_moverz_decoder = ion_moverz_decoder_sin.unsqueeze(1), ion_moverz_decoder_cos.unsqueeze(1)
        intensity_sin, intensity_cos = self.sinusoidal_position_embedding(intensity, self.hidden_size, 1, 1e-7)
        intensity = torch.concat([intensity_sin,intensity_cos],dim=-1)
        return ion_moverz_encoder, ion_moverz_decoder, intensity, spectra_mask_encoder, spectra_mask_decoder

    def tonkenize(self, label_seq, inference_seq):
        fix_seq = [self.aa_dict['<bos>']]
        tgt_seq = []
        label_temp = []
        pre_seq_right = True
        for label_pepblock, inference_pepblock in zip(label_seq.split(' '),inference_seq.split(' ')):
            if len(label_pepblock)>1 or label_pepblock in self.single_aa_mask_list:
                fix_seq += [self.aa_dict['<x>']]
                tgt_seq += [self.aa_dict['<answer>']]+[self.aa_dict[aa] for aa in inference_pepblock]
                
                pre_edge_right = True
                label_temp += [(label_pepblock[0],pre_seq_right)]
                for i in range(len(inference_pepblock)-1):
                    if pre_edge_right and inference_pepblock[i] == label_pepblock[i]:
                        label_temp += [(label_pepblock[i+1],pre_seq_right)]
                    else:
                        pre_edge_right = False
                        pre_seq_right = False
                        label_temp += [('',pre_seq_right)]
                label_temp += [('',pre_seq_right)]

            else: 
                assert self.aa_dict[label_pepblock] == self.aa_dict[inference_pepblock]
                fix_seq += [self.aa_dict[label_pepblock]]
        
        label_temp = [('',pre_seq_right)]*len(fix_seq) + label_temp
        label_temp = label_temp[:-1]
        label = torch.zeros(len(label_temp),21)
        for i, (aa, pre_seq_right) in enumerate(label_temp):
            if aa: label[i,self.tokenize_aa_dict[aa]] = 1
            if pre_seq_right: label[i,-1] = 1

        seq = fix_seq+tgt_seq
        tgt = torch.LongTensor(seq)[:-1]
        tgt = pad(tgt,[0,self.max_len-tgt.shape[0]])
        label = pad(label,[0,0,0,self.max_len-label.shape[0]])
        return label, tgt
    
    def decoder_mask(self, label_seq, inference_seq):
        fix_seq = []
        tgt_seq = []
        for label_pepblock, inference_pepblock in zip(label_seq.split(' '),inference_seq.split(' ')):
            if len(label_pepblock)>1 or label_pepblock in self.single_aa_mask_list:
                fix_seq += [0]
                tgt_seq += [0]+len(inference_pepblock)*[1]
            else: fix_seq += [0]
        
        label_mask = torch.BoolTensor(fix_seq+tgt_seq)
        label_mask = pad(label_mask,[0,self.max_len-len(label_mask)])
        tgt_mask = torch.ones(self.max_len,self.max_len,dtype=bool).triu(diagonal=1)
        tgt_mask[:len(fix_seq)+1,:len(fix_seq)+1] = 0
        tgt_mask = tgt_mask.unsqueeze(0)
        return label_mask, tgt_mask
    
    def token_position(self, seq):
        pepblock_index = [i for i in range(1,len(seq.split(' '))+1)]
        aa_index_in_block = [0]*len(seq.split(' '))
        for i, pepblock in enumerate(seq.split(' '),start=1):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                pepblock_index += [i]+len(pepblock)*[i]
                aa_index_in_block += [0]+[aa_index for aa_index in range(1,len(pepblock)+1)]
        
        pepblock_index = torch.LongTensor([0]+pepblock_index[:-1])
        aa_index_in_block = torch.LongTensor([0]+aa_index_in_block[:-1])
        pepblock_index = pad(pepblock_index,[0,self.max_len-len(pepblock_index)])
        aa_index_in_block = pad(aa_index_in_block,[0,self.max_len-len(aa_index_in_block)])
        return pepblock_index, aa_index_in_block
    
    def peptide_mass_embedding(self, seq, moverz, charge):
        mass = 0
        nterm_mass_tgt = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                nterm_mass_tgt += [mass]
                for aa in pepblock:
                    mass += Residual_seq(aa).mass
                    nterm_mass_tgt += [mass]
            else:
                mass += Residual_seq(pepblock).mass
        
        mass = 0
        nterm_mass_fix = []
        for pepblock in seq.split(' '):
            if len(pepblock)>1 or pepblock in self.single_aa_mask_list:
                mass += Residual_seq(pepblock).mass
                nterm_mass_fix += [mass]
            else:
                mass += Residual_seq(pepblock).mass
                nterm_mass_fix += [mass]
        nterm_mass = nterm_mass_fix+nterm_mass_tgt
        nterm_mass = [0]+nterm_mass[:-1]

        precursor_mass = Ion.precursorion2mass(moverz,charge)
        cterm_mass = precursor_mass - nterm_mass
        nterm_mass = np.pad(nterm_mass,(0,self.max_len-len(nterm_mass)))
        cterm_mass = np.pad(cterm_mass,(0,self.max_len-len(cterm_mass)))
        nterm_mass_1_sin, nterm_mass_1_cos = self.sinusoidal_position_embedding(nterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_1_sin, cterm_mass_1_cos = self.sinusoidal_position_embedding(cterm_mass,   self.hidden_size//self.nhead, 1e4, 1e-5)
        nterm_mass_2_sin, nterm_mass_2_cos = self.sinusoidal_position_embedding(nterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        cterm_mass_2_sin, cterm_mass_2_cos = self.sinusoidal_position_embedding(cterm_mass/2, self.hidden_size//self.nhead, 1e4, 1e-5)
        pe_sin = torch.stack([nterm_mass_1_sin,cterm_mass_1_sin,nterm_mass_2_sin,cterm_mass_2_sin],dim=1).repeat(1,self.nhead//4,1)
        pe_cos = torch.stack([nterm_mass_1_cos,cterm_mass_1_cos,nterm_mass_2_cos,cterm_mass_2_cos],dim=1).repeat(1,self.nhead//4,1)
        return pe_sin, pe_cos

    @staticmethod
    def sinusoidal_position_embedding(mass_position, dim, lambda_max, lambda_min):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*math.pi)
        scale = lambda_min/lambda_max
        div_term = base*scale**(np.arange(0, dim, 2)/dim)
        pe_sin = np.sin(mass_position[...,np.newaxis] / div_term)
        pe_cos = np.cos(mass_position[...,np.newaxis] / div_term)
        return torch.Tensor(pe_sin), torch.Tensor(pe_cos)
