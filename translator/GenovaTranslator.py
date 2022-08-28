import torch
import torch.nn as nn
from GatedAttn import GAU
from ClassicalAttn import DecoderLayer
from torch.cuda.amp import autocast

class PepTokenEmbedding(nn.Module):
    def __init__(self, hidden_size, max_len, aa_dict_size):
        super().__init__()
        self.tgt_token_embedding = nn.Embedding(aa_dict_size, hidden_size)
        self.pepblock_embedding = nn.Embedding(max_len, hidden_size)
        self.aa_embedding = nn.Embedding(max_len, hidden_size)
    
    def forward(self, tgt, pepblock_index, aa_index_in_block):
        tgt = self.tgt_token_embedding(tgt)
        tgt += self.pepblock_embedding(pepblock_index)
        tgt += self.aa_embedding(aa_index_in_block)
        return tgt

class SpectraEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.charge_embedding = nn.Embedding(10, hidden_size)

    def forward(self, src, charge):
        src += self.charge_embedding(charge).unsqueeze(1)
        return src

class GenovaTranslator(nn.Module):
    def __init__(self, hidden_size, n_head, encoder_layer=24, decoder_layer=12, max_len=50*3, aa_dict_size=24) -> None:
        super().__init__()
        self.initial = False
        self.spectra_embedding = SpectraEmbedding(hidden_size)
        self.encoder = nn.ModuleList([GAU(dim=hidden_size) for i in range(encoder_layer)])
        self.norm = nn.LayerNorm(hidden_size)
        self.tgt_embedding = PepTokenEmbedding(hidden_size, max_len, aa_dict_size)
        self.decoder = nn.ModuleList([DecoderLayer(hidden_size,n_head,0.1) for i in range(decoder_layer)])
        self.output = nn.Linear(hidden_size,21)

    def forward(self, src, tgt, tgt_mask, encoder_input, crossattn):
        src = self.spectra_embedding(**src)
        tgt = self.tgt_embedding(**tgt)
        for l_encoder in self.encoder:
            src = l_encoder(src, **encoder_input)
        mem = self.norm(src)
        for l_decoder in self.decoder:
            tgt = l_decoder(tgt, tgt_mask=tgt_mask, mem=mem, **crossattn)
        tgt = self.output(tgt)
        return tgt

    @autocast()
    @torch.no_grad()
    def mem_get(self, src, encoder_input):
        src = self.spectra_embedding(**src)
        for l_encoder in self.encoder:
            src = l_encoder(src, **encoder_input)
        mem = self.norm(src)
        return mem
    
    @autocast()
    @torch.no_grad()
    def inference_step(self, tgt, tgt_mask, mem=None, past_tgts=None, past_mems=None, pep_mass=None, mem_key_padding_mask=None, peaks_moverz=None):
        past_tgt_list = []
        tgt = self.tgt_embedding(**tgt)
        if self.initial:
            past_mem_list = []
            for l_decoder in self.decoder:
                tgt, past_tgt, past_mem = l_decoder(tgt, tgt_mask=tgt_mask, mem=mem, pep_mass=pep_mass, mem_key_padding_mask=mem_key_padding_mask, peaks_moverz=peaks_moverz)
                past_tgt_list.append(past_tgt)
                past_mem_list.append(past_mem)
            return past_tgt_list, past_mem_list
        else:
            for past_tgt, past_mem, l_decoder in zip(past_tgts, past_mems, self.decoder):
                tgt, past_tgt_new = l_decoder(tgt, tgt_mask=tgt_mask, past_tgt=past_tgt, past_mem=past_mem, pep_mass=pep_mass, mem_key_padding_mask=mem_key_padding_mask)
                past_tgt_list.append(past_tgt_new)
            tgt = self.output(tgt)
            return tgt, past_tgt_list

    def inference_initial(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("inference_initial mode is expected to be boolean")
        self.initial = mode
        for module in self.decoder:
            if hasattr(module, "inference_initial"): module.inference_initial(mode)

    def inference_iter(self):
        self.inference_initial(False)