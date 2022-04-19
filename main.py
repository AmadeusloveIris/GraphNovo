import hydra
import wandb
import genova
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from genova.utils.BasicClass import Residual_seq

from itertools import combinations_with_replacement
aa_datablock_dict = {}
aalist = Residual_seq.output_aalist()
for num in range(1,7):
    for i in combinations_with_replacement(aalist,num):
        aa_datablock_dict[i] = Residual_seq(i).mass

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig)->None:
    spec_header = pd.read_csv('/home/z37mao/genova_dataset_index.csv',low_memory=False,index_col='Spec Index')
    spec_header = spec_header[spec_header['MSGP File Name']=='1_3.msgp']
    task = genova.task.Task(cfg,'/home/z37mao/Genova/save', aa_datablock_dict=aa_datablock_dict, distributed=True)
    task.initialize(spec_header,'/home/z37mao/',spec_header,'/home/z37mao/')
    for loss_train, total_step in task.train():
        loss_eval = task.eval()
        wandb.log({'train_loss':loss_train, 'eval_loss': loss_eval}, step=total_step)

if __name__=='__main__':
    main()