
# RL-ASM
This repository is the PyTorch implementation of "Approximate Subgraph Matching with Reinforcement Learning". For more details, please refer to our paper.
The Architecture of our model is as follows:
<p align="center">
    <image src="framework_0516.jpg" width="1000" />
</p>

## Installation

Package requirements are in `environment.yml`. Please install required packages and versions. 

### Conda Setup

To build and run the conda environment use the commands below:
```
conda env create -f environment.yml
conda activate RL_ASM_env
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

## Data Preparation
We have provided Datasets [here](https://drive.google.com/drive/folders/181SDISsNurT_w3eO5cgR7uJof5f66LRS?usp=drive_link). Unzip these files and move them to ```data```.

Alternatively, you can obtain the raw datasets from this [site](https://chrsmrrs.github.io/datasets/docs/datasets/) and this [site](https://snap.stanford.edu/data/email-EuAll.html). To prepare the datasets, process the raw data using the following commands:

```
python ./uclasm/matching/create_graph_pairs.py
python ./uclasm/matching/data_sampling_pipeline.py
```

## Run
The configuration files for imitation learning and PPO are stored in the  ```config``` folder.
### Training

To run imitation learning, use the following command:
```
python ./uclasm/matching/PG_matching_imitation_learning_undirect_batch.py
```
Fine-tune the model with PPO:
```
python ./uclasm/matching/PG_matching_RL_PPO_refactor.py
```

### Testing
A trained model is available at  ```ckpt_RL\MSRC_21_RL.pth```. You can test this model with the following command:
```
python ./uclasm/matching/PG_test_RL_iteration.py
```

### Analyzing Results
Experiment results are stored in the ```results``` folder. You can analyze these results using:
```
python ./uclasm/matching/analyze_test_records.py
```