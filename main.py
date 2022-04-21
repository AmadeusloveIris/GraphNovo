import hydra
import wandb
import genova
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from genova.utils.BasicClass import Residual_seq

from itertools import combinations_with_replacement
aa_datablock_dict = {}
aalist = Residual_seq.output_aalist()
for num in range(1,7):
    for i in combinations_with_replacement(aalist,num):
        aa_datablock_dict[i] = Residual_seq(i).mass

def init_wandb(cfg):
    run = wandb.init(entity=cfg.wandb.entity, 
            project=cfg.wandb.project, name=cfg.wandb.name, 
            config=OmegaConf.to_container(cfg))
    return run

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    train_spec_header = pd.read_csv('/data/genova_data_filted/genova_filted_dataset_index.csv',low_memory=False,index_col='Spec Index')
    eval_spec_header = pd.read_csv('/data/genova_data/genova_dataset_index.csv',low_memory=False,index_col='Spec Index')
    eval_spec_header = eval_spec_header[eval_spec_header['Experiment Name']=='Plasma']
    task = genova.task.Task(cfg,serialized_model_path=cfg.train.serialized_model_path,aa_datablock_dict=aa_datablock_dict)
    task.initialize(train_spec_header=train_spec_header,train_dataset_dir='/data/genova_data_filted',val_spec_header=eval_spec_header,val_dataset_dir='/data/genova_data')
    if dist.is_initialized() and dist.get_rank()==0: run = init_wandb(cfg)
    best_loss = float('inf')
    for loss_train, total_step, epoch in task.train():
        if dist.get_rank()==0: run.log({'train_loss':loss_train}, step=total_step)
        if total_step%cfg.train.eval_period==0:
            loss_eval = task.eval()
            if dist.get_rank()==0:
                print('epoch:{}, step:{}, train loss:{}, eval loss:{}'.format(epoch,total_step,loss_train,loss_eval))
                run.log({'eval_loss': loss_eval}, step=total_step)
                if best_loss>loss_eval:
                    best_loss = loss_eval
                    task.model_save()
            dist.barrier()
    run.finish()

if __name__=='__main__':
    main()
