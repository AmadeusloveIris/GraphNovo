import hydra
import wandb
import genova
import numpy as np
import pandas as pd
import torch.distributed as dist
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    
    #需要修改为输入变量
    if cfg.mode == 'train':
        assert cfg.train.serialized_model_path==None or cfg.train_spec_header_path==None \
        or cfg.eval_spec_header_path==None or cfg.train_dataset_dir==None or cfg.eval_dataset_dir==None
        train_spec_header = pd.read_csv(cfg.train_spec_header_path,index_col='Spec Index')
        eval_spec_header = pd.read_csv(cfg.eval_spec_header_path,index_col='Spec Index')
        task = genova.task.Task(cfg,serialized_model_path=cfg.train.serialized_model_path, distributed=cfg.dist)
        task.initialize(train_spec_header=train_spec_header,train_dataset_dir=cfg.train_dataset_dir,val_spec_header=eval_spec_header,val_dataset_dir=cfg.eval_dataset_dir)
        
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
                    if (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                        print('epoch:{}, step:{}, train loss:{}, eval loss:{}, accuracy: {}, recall: {}, precision: {}'.format(epoch,total_step,loss_train,loss_eval,accuracy,recall,precision))
                        wandb.log({'train_loss': loss_train, 'eval_loss': loss_eval, 'accuracy': accuracy, 'recall': recall, 'precision': precision}, step=total_step)
                        if best_loss>loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                    print('step: {}, train_loss: {}'.format(total_step,loss_train)) 
                    wandb.log({'train_loss':loss_train}, step=total_step)
            run.finish()
        else:
            for loss_train, total_step, epoch in task.train():
                if total_step%cfg.train.eval_period==0:
                    loss_eval = task.eval()
                    if (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()):
                        print('epoch:{}, step:{}, train loss:{}, eval loss:{}'.format(epoch,total_step,loss_train,loss_eval))
                        wandb.log({'train_loss': loss_train,'eval_loss': loss_eval}, step=total_step)
                        if best_loss>loss_eval:
                            best_loss = loss_eval
                            task.model_save()
                elif (dist.is_initialized() and dist.get_rank()==0) or (not dist.is_initialized()): 
                    print('step: {}, train_loss: {}'.format(total_step,loss_train))
                    wandb.log({'train_loss':loss_train}, step=total_step)
            run.finish()
    elif cfg.mode == 'inference':
        if cfg.infer.testset == 'C_Elegans':
            preprocess_file = 'C_Elegan.csv'
        elif cfg.infer.testset == 'A_Thaliana':
            preprocess_file = 'A_Thaliana.csv'
        elif cfg.infer.testset == 'E_Coli':
            preprocess_file = 'E_Coli.csv'
            
        genova_dir = get_original_cwd()
        print('genova_dir: ', genova_dir)
        spec_header = pd.read_csv(genova_dir+cfg.infer.data_dir+preprocess_file, index_col='Spec Index')
        print('Full dataset shape: ', spec_header.shape)
        print('MSGP File Name list: ', set(spec_header['MSGP File Name'].tolist()))

        # initialization
        task = genova.task.Task(cfg,genova_dir+'/save', distributed=False)
        task.test_initialize(test_spec_header=spec_header,test_dataset_dir=genova_dir+cfg.infer.data_dir)
        task.inference()

if __name__=='__main__':
    main()
