import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/ISM_custom/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
from matching.environment import environment,get_init_action
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from PG_matching import policy_network
with open('./data/Email_trainset_dens_0.2_n_8_num_2000_10_05.pkl','rb') as f:
    train_dataset = pickle.load(f)
with open('./data/Email_testset_dens_0.2_n_8_num_50_10_05.pkl','rb') as f:
    test_dataset = pickle.load(f)
test_dataset.gs[0].nxgraph.init_x = train_dataset.gs[0].nxgraph.init_x

with open('./data/Email_testset_dens_0.2_n_8_num_50_10_05_new.pkl','wb') as f:
    pickle.dump(test_dataset,f)
   