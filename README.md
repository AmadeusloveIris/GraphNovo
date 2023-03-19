# GraphNovo:

**Mitigating the missing fragmentation problem in _de novo_ peptide sequencing with a two stage graph-based deep learning model**

Author: Zeping Mao(z37mao@uwaterloo.ca), Ruixue Zhang(r267zhan@uwaterloo.ca)

Data repository: <https://drive.google.com/drive/folders/18KnMWPoTsMporY2N4ttECXzi5V--NSZx?usp=sharing>

## Usage
This project uses hydra to manage configure file and use wandb to visiualize training step. If you are not familiar with them, please go to: <https://hydra.cc>(hydra) and <https://wandb.ai>(Weights & Biases).

### Hardware Requirement
For model training, we suggests utilizing an 80GB A100 GPU.

For model inferencing, our tests have shown that most spectra perform well under 16GB V100 GPU. However if the size of spectrum is too large, we recommend using a GPU with higher memory or performing inferencing on a CPU with enough RAM instead.

In case you plan to build a graph from scratch, ensure that you have access to a server with over 500GB of RAM.

### Setup
For envrionment prepare, please use conda:
~~~
conda env create -f environment.yml
~~~

First build cython modules and all possible peptide mass(within 1000 Da) rainbow table.

~~~
sh setup.sh
~~~

Before your first time using, you need to set up wandb account. For details about how to set up a wandb account, please read: <https://docs.wandb.ai/quickstart>

------
### Train
If you want to use constructed graph to train from scrach, you need to use following command: (we assume that you download data under GraphNovo file fold. If not, please adjust path by yourself.)
~~~
python main.py serialized_model_path=save/ckpt dist=false task=optimal_path train_spec_header_path=training_dataset/preprocessed/training_dataset.csv eval_spec_header_path=validation_dataset/preprocessed/validation_dataset.csv train_dataset_dir=training_dataset/preprocessed/ eval_dataset_dir=validation_dataset/preprocessed/ wandb.project=GraphNovo wandb.name=PathSearcher
~~~
For 'task' argument, you can choose 'optimal_path' or 'sequence_generation'. Use 'optimal_path' will train a GraphNovo_PathSearcher and 'sequence_generation' will train GraphNovo_SeqFiller.

By the way, if you want to train model on multiple GPUs. Please use:
~~~
torchrun --<other arguments for torchrun> main.py <some path> dist=True <other argument>
~~~

Persistent model is named as <wandb.project>_<wandb.name>. If you want to use anathor name, please remember to change these arguments.

------
### Inference
#### PathSearcher
Please make a directory of 'graphnovo_data' and 'prediction'. Download 'overall' or 'barrier' data under the directory 'graphnovo_data'. All prediction will be saved under 'prediction'. Test set can be 'A_Thaliana' for A. thaliana, 'C_Elegans' for C. elegans, or 'E_Coli' for E. coli.

To generate optimal path of A. thaliana, please use:
~~~
python main.py mode=inference serialized_model_path=save/ckpt dist=false infer=optimal_path_inference task=optimal_path wandb.project=GraphNovo wandb.name=PathSearcher infer.beam_size=20 infer.testset=A_Thaliana infer.optimal_path_file=prediction/optimal_path/A_Thaliana_beam20_sum.csv infer.data_dir=graphnovo_data/overall
~~~

The predicted optimal path will be stored under './prediction/optimal_path'. wandb.project and wandb.name should be consistent with the training setting to use the correct checkpoint. 

#### SeqFiller
To generate the final prediction of peptide sequence of A. thaliana, please use:
~~~
python main.py mode=inference serialized_model_path=save/ckpt dist=false infer=sequence_generation_inference task=sequence_generation wandb.project=GraphNovo wandb.name=SeqFiller infer.beam_size=20 infer.testset=A_Thaliana infer.optimal_path_file=prediction/optimal_path/A_Thaliana_beam20_sum.csv infer.output_file=prediction/sequence_generation/A_Thaliana_beam20_sum_beam20_sum.csv infer.data_dir=graphnovo_data/overall
~~~

The predicted peptide sequence is generated along the optimal path saved in 'prediction/optimal_path/A_Thaliana_beam20_sum.csv' for this example. The final result is saved under 'prediction/sequence_generation'. 

------
### Graph Construction
For graph construction, please prepare 50G memory per thread. If you want to construct the graph, please remember to put csv file(psm header) and mgf file to the same folder.
~~~
cd genova/utils
sh parrllel_preprocessing.sh <total_threads> <data_path> <psm header name without 'csv'>
~~~

After all graph_constructor finished, constructed graph will be stored at same location as 'data_path'. Remember to concat all generated csv files.


