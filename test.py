import wandb
import hydra
from omegaconf import OmegaConf
@hydra.main(config_path="configs", config_name='config.yaml')
def config_print(cfg):
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    print(OmegaConf.to_yaml(cfg))
    print()
    print(OmegaConf.to_yaml(cfg.encoder))

if __name__ == '__main__':
    config_print()