# RL-ASM
This repository is the PyTorch implementation of "Approximate Subgraph Matching with Reinforcement Learning" . For more details, please refer to our paper.
The Architecture of our model is as follows:
 <p align =“center”>
    <image  src=framework_0516.jpg width=1000 />
 </p>

## Installation

Package requirements are in `environment.yml`. Please install required packages and versions. 

### Conda Setup

To build and run the conda environment use the commands below:
```
conda env create -f environment.yml
```

## Datasets

Get the raw datasets from: https://chrsmrrs.github.io/datasets/docs/datasets/ and https://snap.stanford.edu/data/email-EuAll.html
Create the train, valid and test set with the below command:
'''
pyhton ./uclasm/matching/data_sampling_pipeline.py
'''

## Run

### Testing
Run the below command:
```
python ./uclasm/matching/PG_test_RL_iteration.py
```
### Training

Run Imitaiton learning with the command:
```
pyhton ./uclasm/matching/PG_matching_imitation_learning_undirect_batch.py
```
Fine-tune the model with PPO:
```
pyhton ./uclasm/matching/PG_matching_RL_PPO_refactor.py
```

