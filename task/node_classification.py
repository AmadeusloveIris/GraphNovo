import os
import gzip
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import train_test_split

import genova
from genova.data.dataset import GenovaDataset
from genova.data.collator import GenovaCollator
from genova.data.sampler import GenovaSampler
from genova.train_utils.ckpt import save_ckp, load_ckp

import wandb
import hydra
from hydra.utils import get_original_cwd
import omegaconf

from typing import Iterator, Optional

class NodeClassification:
    def __init__(self, cfg, train_dir, local_rank, ngpus_per_node):
        self.cfg = cfg
        self.local_rank = local_rank
        self.ngpus_per_node = ngpus_per_node
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        self.device = torch.device("cuda", local_rank)

        self.train_dir = train_dir
        self.model = genova.GenovaEncoder(self.cfg, bin_classification=True).to(local_rank)
        if torch.distributed.is_initialized():
            print('Distributing computing started...')
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4)
        self.scaler = GradScaler()
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.epoch = 0

        if len(os.listdir(os.path.join(self.train_dir, 'checkpoint'))) != 0:
            self.model, self.optimizer, self.epoch = load_ckp(self.train_dir, self.model, self.optimizer)

    def load_train_data(self, spec_header, dataset_dir_path, train_prop=0.9):
        train_size = int(train_prop * len(spec_header))
        val_size = len(spec_header) - train_size
        if self.local_rank == 0:
            print("train_size / val_size: ", train_size, " / ", val_size)

        # import pdb; pdb.set_trace()
        spec_header_train, spec_header_val = train_test_split(spec_header, test_size=val_size)
        spec_header_train = spec_header_train[:200]
        spec_header_val = spec_header_val[:50]

        train_dataset = GenovaDataset(self.cfg, spec_header=spec_header_train, dataset_dir_path=dataset_dir_path)
        val_dataset = GenovaDataset(self.cfg, spec_header=spec_header_val, dataset_dir_path=dataset_dir_path)
        train_dataset = Subset(train_dataset, np.arange(1000))
        val_dataset = Subset(val_dataset, np.arange(300))

        collate_fn = GenovaCollator(self.cfg)
        # print('train local rank:', self.local_rank)
        train_sampler = GenovaSampler(train_dataset, self.cfg, 13, 2, 0.5)
        # train_sampler = DistributedSamplerWrapper(train_sampler, num_replicas=self.ngpus_per_node, rank=self.local_rank)
        # print('val local rank:', self.local_rank)
        val_sampler = GenovaSampler(val_dataset, self.cfg, 13, 2, 0.5)
        # val_sampler = DistributedSamplerWrapper(val_sampler, num_replicas=self.ngpus_per_node, rank=self.local_rank)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=1)

        return train_loader, val_loader

    def load_test_data(self, spec_header, dataset_dir_path):
        print("test_size : ", spec_header.shape[0])
        test_dataset = GenovaDataset(self.cfg, spec_header=spec_header, dataset_dir_path=dataset_dir_path)
        test_dataset = Subset(test_dataset, np.arange(600))
        collate_fn = GenovaCollator(self.cfg)
        test_sampler = GenovaSampler(test_dataset, self.cfg, 13, 2, 0.5)

        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn, num_workers=1)
        print(len(test_dataset))

        return test_loader

    def train(self, train_loader, val_loader=None):
        detect_period = 50
        train_loss, train_tp_total, train_tn_total, train_fp_total, train_fn_total = 0, 0, 0, 0, 0
        accuracy, recall, precision = 0, 0, 0
        best_loss = float("inf")
        is_best = False

        for epoch in range(self.epoch, 1):
            if self.local_rank == 0:
                print('Epoch:', epoch)
            for i, (encoder_input, labels, node_mask) in enumerate(train_loader):
                # print('train rank: ', self.local_rank, "i: ", i)

                encoder_input = self.encoder_input_cuda(encoder_input)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                with autocast():
                    self.model.train()
                    output = self.model(**encoder_input)
                    loss = self.loss_fn(output[~node_mask], labels[~node_mask])

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # import pdb; pdb.set_trace()
                output = torch.argmax(output[~node_mask], -1)
                labels = labels[~node_mask]

                labels_gather_list = [torch.zeros_like(labels) for _ in range(self.ngpus_per_node)]
                torch.distributed.all_gather(labels_gather_list, labels)
                labels = torch.cat(labels_gather_list, dim=0)

                output_gather_list = [torch.zeros_like(output) for _ in range(self.ngpus_per_node)]
                torch.distributed.all_gather(output_gather_list, output)
                output = torch.cat(output_gather_list, dim=0)

                train_tp = ((output == labels)[output == 1]).sum()
                train_tp_total += train_tp
                train_tn = ((output == labels)[output == 0]).sum()
                train_tn_total += train_tn
                train_fp = ((output != labels)[output == 1]).sum()
                train_fp_total += train_fp
                train_fn = ((output != labels)[output == 0]).sum()
                train_fn_total += train_fn
                train_loss += loss.item()

                accuracy += (train_tp + train_tn) / (train_tp + train_tn + train_fp + train_fn)
                recall += train_tp / (train_tp + train_fn)
                precision += train_tp / (train_tp + train_fp)

                if i % detect_period == 0 and i > 0:
                    if self.local_rank == 0:
                        print('Train batch: ', i)
                        train_total = train_tp_total+train_tn_total+train_fp_total+train_fn_total
                        wandb.log({"train_loss": train_loss / train_total,
                                   "train_accuracy_by_label": (train_tp_total+train_tn_total) / train_total,
                                   "train_recall_by_label": train_tp_total / (train_tp_total+train_fn_total),
                                   "train_precision_by_label": train_tp_total / (train_tp_total+train_fp_total),
                                   "train_accuracy_by_batch": accuracy / detect_period,
                                   "train_recall_by_batch": recall / detect_period,
                                   "train_precision_by_batch": precision / detect_period}
                                  )
                        train_loss, train_tp_total, train_tn_total, train_fp_total, train_fn_total = 0, 0, 0, 0, 0

                    # dist.barrier()
                    if val_loader is not None:
                        val_loss, _ = self.eval(val_loader)
                        if val_loss < best_loss:
                            is_best = True
                            best_loss = val_loss

                        checkpoint = {'epoch': epoch,
                                      'model_state_dict': self.model.state_dict(),
                                      'optimizer_state_dict': self.optimizer.state_dict()}
                        save_ckp(checkpoint, is_best, self.train_dir)

                # dist.barrier()

    def eval(self, val_loader):
        detect_period = 50

        self.model.eval()
        with torch.no_grad():
            val_loss, val_tp_total, val_tn_total, val_fp_total, val_fn_total = 0, 0, 0, 0, 0
            val_accuracy, val_recall, val_precision = 0, 0, 0

            for i, (val_encoder_input, val_labels, val_node_mask) in enumerate(val_loader):
                if i % detect_period == 0:
                    print('Validation batch: ', i)

                val_encoder_input = self.encoder_input_cuda(val_encoder_input)
                val_labels = val_labels.to(self.device)
                self.model.eval()
                val_output = self.model(**val_encoder_input)
                loss = self.loss_fn(val_output[~val_node_mask], val_labels[~val_node_mask])
                val_output = torch.argmax(val_output[~val_node_mask], -1)
                val_labels = val_labels[~val_node_mask]

                # val_labels_gather_list = [torch.zeros_like(val_labels) for _ in range(self.ngpus_per_node)]
                # torch.distributed.all_gather(val_labels_gather_list, val_labels)
                # val_labels = torch.cat(val_labels_gather_list, dim=0)
                #
                # val_output_gather_list = [torch.zeros_like(val_output) for _ in range(self.ngpus_per_node)]
                # torch.distributed.all_gather(val_output_gather_list, val_output)
                # val_output = torch.cat(val_output_gather_list, dim=0)

                val_tp = ((val_output == val_labels)[val_output == 1]).sum()
                val_tp_total += val_tp
                val_tn = ((val_output == val_labels)[val_output == 0]).sum()
                val_tn_total += val_tn
                val_fp = ((val_output != val_labels)[val_output == 1]).sum()
                val_fp_total += val_fp
                val_fn = ((val_output != val_labels)[val_output == 0]).sum()
                val_fn_total += val_fn
                val_loss += loss.item()

                val_accuracy += (val_tp+val_tn) / (val_tp+val_tn+val_fp+val_fn)
                val_recall += val_tp / (val_tp+val_fn)
                val_precision += val_tp / (val_tp+val_fp)

            val_total = val_tp_total + val_tn_total + val_fp_total + val_fn_total

            if self.local_rank == 0:
                wandb.log({"val_loss": val_loss / val_total,
                           "val_accuracy_by_label": (val_tp_total + val_tn_total) / val_total,
                           "val_recall_by_label": val_tp_total / (val_tp_total + val_fn_total),
                           "val_precision_by_label": val_tp_total / (val_tp_total + val_fp_total),
                           "val_accuracy_by_batch": val_accuracy / (i + 1),
                           "val_recall_by_batch": val_recall / (i + 1),
                           "val_precision_by_batch": val_precision / (i + 1)}
                          )

            return val_loss, val_output

    def encoder_input_cuda(self, encoder_input):
        for section_key in encoder_input:
            if isinstance(encoder_input[section_key], torch.Tensor):
                encoder_input[section_key] = encoder_input[section_key].to(self.device)
                continue
            for key in encoder_input[section_key]:
                if isinstance(encoder_input[section_key][key], torch.Tensor):
                    encoder_input[section_key][key] = encoder_input[section_key][key].to(self.device)
        return encoder_input


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: omegaconf.DictConfig):
    local_rank = int(os.environ['LOCAL_RANK'])

    ngpus_per_node = 4

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if local_rank == 0:
        wandb.init(project="Genova", entity="rxnatalie", settings=wandb.Settings(start_method="thread"))

    mode = "train"

    original_path = get_original_cwd()
    spec_header = pd.read_csv(os.path.join(original_path, 'pretrain_data_sparse/genova_psm.csv'), index_col='index')
    spec_header = spec_header[
        np.logical_or(spec_header['Experiment Name'] == 'Cerebellum', spec_header['Experiment Name'] == 'HeLa')]
    small_spec = spec_header[spec_header['Node Number'] <= 256]

    DIR_PATH = os.path.join(original_path, 'save/test1')
    if not os.path.exists(os.path.join(DIR_PATH, 'checkpoint')):
        print('Creating dirs...')
        os.makedirs(os.path.join(DIR_PATH, 'checkpoint'))

    node_classification = NodeClassification(cfg.model, DIR_PATH, local_rank, ngpus_per_node)

    if mode == "train":
        train_loader, val_loader = node_classification.load_train_data(spec_header=small_spec,
                                    dataset_dir_path=os.path.join(original_path, 'pretrain_data_sparse'))

        if local_rank == 0:
            print('Start training...')
        node_classification.train(train_loader, val_loader)

    elif mode == "test":
        test_loader = node_classification.load_test_data(spec_header=small_spec,
                        dataset_dir_path=os.path.join(original_path, 'pretrain_data_sparse'))
        if local_rank == 0:
            print('Start inferencing...')
        node_classification.eval(test_loader)


if __name__ == '__main__':
    main()