import os
import hydra
import torch
import genova
from omegaconf import open_dict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig, OmegaConf

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
rank=dist.get_rank()*torch.ones(1)

rank = rank.to(local_rank)

all_rank = dist.reduce(rank,0)

if dist.get_rank()==0:
    print("all_rank:",all_rank)
    print("rank:",rank)

#print(os.environ["LOCAL_WORLD_SIZE"])
