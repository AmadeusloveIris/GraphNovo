import os
import hydra
import torch
import genova
from omegaconf import open_dict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()