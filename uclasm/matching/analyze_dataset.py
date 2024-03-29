import pickle
import time
import torch.nn.functional as F
import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/") 
with open('data/SYNTHETIC/SYNTHETIC_testset_sparse_noiseratio_0_n_16_32_num_01_31_RWSE.pkl','rb') as f:
    test_dataset = pickle.load(f)

pairs = test_dataset.pairs

node_num_g1 = []
edge_num_g1 = []
node_num_g2 = []
edge_num_g2 = []

for pair,value in pairs.items():
    g1 = test_dataset.look_up_graph_by_gid(pair[0]).get_nxgraph()
    g2 = test_dataset.look_up_graph_by_gid(pair[1]).get_nxgraph()
    node_num_g1.append(g1.number_of_nodes())
    node_num_g2.append(g2.number_of_nodes())
    edge_num_g1.append(g1.number_of_edges())
    edge_num_g2.append(g2.number_of_edges())

print(sum(node_num_g1)/len(node_num_g1))
print(sum(node_num_g2)/len(node_num_g2))
print(sum(edge_num_g1)/len(edge_num_g1))
print(sum(edge_num_g2)/len(edge_num_g2))




