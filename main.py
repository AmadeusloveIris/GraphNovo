import hydra
import wandb
import genova
import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    train_spec_header = pd.read_csv('/data/genova_data/genova_dataset_index.csv',low_memory=False,index_col='Spec Index')
    train_spec_header = train_spec_header[np.logical_or(train_spec_header['Experiment Name']=='Cerebellum',train_spec_header['Experiment Name']=='Hela')]
    train_spec_header = train_spec_header[train_spec_header['Node Number']<=512]
    train_spec_header = train_spec_header[np.logical_and(train_spec_header['Relation Num']<=5e5,train_spec_header['Edge Num']<=1e6)]
    eval_spec_header = pd.read_csv('/data/genova_data/genova_dataset_index.csv',low_memory=False,index_col='Spec Index')
    eval_spec_header = eval_spec_header[eval_spec_header['Experiment Name']=='Plasma']
    eval_spec_header = eval_spec_header[eval_spec_header['Annotated Sequence'].str.len()<=32]
    task = genova.task.Task(cfg,serialized_model_path=cfg.train.serialized_model_path)
    task.initialize(train_spec_header=train_spec_header,train_dataset_dir='/data/genova_data',val_spec_header=eval_spec_header,val_dataset_dir='/data/genova_data')
    if dist.is_initialized() and dist.get_rank()!=0: pass
    else:
        run = wandb.init(entity=cfg.wandb.entity, 
                         project=cfg.wandb.project, name=cfg.wandb.name, 
                         config=OmegaConf.to_container(cfg))
        wandb.watch(task.model, criterion=task.eval_loss_fn, log='all', log_freq=cfg.train.eval_period*cfg.train.detect_period, log_graph=True)
    best_loss = float('inf')
    if cfg.task =='node_classification':
        for loss_train, total_step, epoch in task.train():
            if total_step%cfg.train.eval_period==0:
                loss_eval, accuracy, recall, precision = task.eval()
                task.scheduler.step(loss_eval)
                if dist.get_rank()==0:
                    print('epoch:{}, step:{}, train loss:{}, eval loss:{}, accuracy: {}, recall: {}, precision: {}'.format(epoch,total_step,loss_train,loss_eval,accuracy,recall,precision))
                    wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, 'accuracy': accuracy, 'recall': recall, 'precision': precision}, step=total_step)
                    if best_loss>loss_eval:
                        best_loss = loss_eval
                        task.model_save()
                dist.barrier()
            elif dist.get_rank()==0:
                print('step: {}, train_loss: {}'.format(total_step,loss_train)) 
                wandb.log({'train_loss':loss_train}, step=total_step)
        run.finish()
    else:
        for loss_train, total_step, epoch in task.train():
            if total_step%cfg.train.eval_period==0:
                loss_eval = task.eval()
                if dist.get_rank()==0:
                    print('epoch:{}, step:{}, train loss:{}, eval loss:{}'.format(epoch,total_step,loss_train,loss_eval))
                    wandb.log({'train_loss': loss_train,'eval_loss': loss_eval}, step=total_step)
                    if best_loss>loss_eval:
                        best_loss = loss_eval
                        task.model_save()
            elif dist.get_rank()==0: 
                print('step: {}, train_loss: {}'.format(total_step,loss_train))
                wandb.log({'train_loss':loss_train}, step=total_step)
        run.finish()

if __name__=='__main__':
    main()
