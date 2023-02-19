# GraphNovo:

**Mitigating the missing fragmentation problem in _de novo_ peptide sequencing with a two stage graph-based deep learning model**

Author: Zeping Mao(z37mao@uwaterloo.ca), Ruixue Zhang(r267zhan@uwaterloo.ca)

Data repository: <https://drive.google.com/drive/folders/18KnMWPoTsMporY2N4ttECXzi5V--NSZx?usp=sharing>

## Usage
This project use hydra to manage configure file and use wandb to visiualize training step. If you are not familiar with them, please go to: <https://hydra.cc>(hydra) and <https://wandb.ai>(Weights & Biases)

### Setup

First build cython modules and all possible peptide mass(within 1000 Da) rainbow table.

~~~
sh setup.sh
~~~

Before your first time using, you need to set up wandb account. For details about how to set up a wandb account, please read: <https://docs.wandb.ai/quickstart>

### Train
If you want to use constructed graph to train from scrach, you need to use following command: (we assume that you download data under GraphNovo file fold. If not, please adjust path by yourself.)
~~~
python main.py serialized_model_path=save/ckpt dist=false task=optimal_path train_spec_header_path=training_dataset/preprocessed/training_dataset.csv eval_spec_header_path=validation_dataset/preprocessed/validation_dataset.csv train_dataset_dir=training_dataset/preprocessed/ eval_dataset_dir=validation_dataset/preprocessed/ wandb.project=GraphNovo wandb.name=PathSearcher
~~~
For 'task' argument, you can choose 'optimal_path' or 'sequence_generation'. Use 'optimal_path' will train a GraphNovo_PathSearcher and 'sequence_generation' will train GraphNovo_SeqFiller

By the way, if you want to training model on multiple GPUs, Please use:
~~~
torchrun --<other arguments for torchrun> main.py <some path> dist=True <other argument>
~~~

Persistent model are named as <wandb.project>_<wandb.name> So if you want to use anathor name, please remember to change these arguments.

### Inference

### Graph Construction
For graph construction, please prepare 50G memory per thread. If you want to construct the graph. Please remember to put csv file(psm header) and mgf file to same fold.
~~~
cd genova/utils
sh parrllel_preprocessing.sh <total_threads> <data_path> <psm header name without 'csv'>
~~~

After all graph_constructor finished, constructed graph will be stored at same location as 'data_path', remember concat all generated csv file.
