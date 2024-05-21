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
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/uclasm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/GraphGPS/") 

from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth
from PG_matching_RL_PPO import _create_batch_data,_preprocess_NSUBS,_create_model


import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
# from PG_structure import update_state
import random
import gc
import datetime

from environment import environment,get_init_action,calculate_cost
import sys
import torch_geometric
torch_geometric.seed_everything(1)







device = FLAGS.device
def clear_directory(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录
        except Exception as e:
            print(f"删除 {file_path} 失败。原因: {e}")

if FLAGS.dataset == 'SYNTHETIC':
    dim = 13
    num_nodes = '16_32'
if FLAGS.dataset == 'AIDS':
    dim = 40
    num_nodes = '6_10'
if FLAGS.dataset == 'EMAIL':
    dim = 47
    num_nodes = '16_32'
if FLAGS.dataset == 'MSRC_21':
    dim = 27
    num_nodes = '16_32'



model = _create_model(dim)
def test_checkpoint_model(ckpt_pth,test_dataset):
     with torch.no_grad():
    
        checkpoint = torch.load(ckpt_pth,map_location=torch.device(FLAGS.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        env = environment(test_dataset)
        costs = []
        for episode in range(1000):
        # for episode in range(len(test_dataset.pairs.keys())):
            state_init = env.reset()
            stack = [state_init]
            while stack:
                state = stack.pop()
                state.action_space = state.get_action_space(env.order)
                pre_processed = _preprocess_NSUBS(state,0)
                batch, data = _create_batch_data([pre_processed])
                batch=batch[:-1]
               
                out_policy, out_value = \
                    model(*batch,
                        True,
                        graph_filter=None, filter_key=None,
                    )
                action_prob = F.softmax(out_policy[0] - out_policy[0].max()) + 1e-10
                _,action_ind = action_prob.max(0)
                action = state.action_space[action_ind]
                newstate, state,reward, done = env.step(state,action)
                
                stack.append(newstate)
 
                if done:
                    costs.append(calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping))
                    model.reset_cache()
                    break
            

        # 返回总奖励，以便在后续可能使用
        # print(sum(costs)/len(costs))
        return sum(costs)/len(costs)


if FLAGS.noiseratio == 0:
    noiseratio = '_noiseratio_0'
elif FLAGS.noiseratio == 5:
     noiseratio = '_noiseratio_5.0'
elif FLAGS.noiseratio == 10:
     noiseratio = '_noiseratio_10.0'

# 使用该函数测试多个检查点
testset = f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/data/{FLAGS.dataset}/{FLAGS.dataset}_testset_dense{noiseratio}_n_{num_nodes}_num_01_31_LapPE.pkl'
# testset = './data/SYNTHETIC/SYNTHETIC_trainset_dense_noiseratio_0_5_10_n_16_32_num_01_31_RWSE.pkl'
with open(testset,'rb') as f:
    test_dataset = pickle.load(f)

time = FLAGS.modelID
try:
    clear_directory(f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/runs_RL_test/{time}_{FLAGS.noiseratio}/')
except:
    pass
writer = SummaryWriter(f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/runs_RL_test/{time}_{FLAGS.noiseratio}/')
for i in range(0, 1200000, 2000):
    checkpoint = f'/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/{FLAGS.ckpt_folder}/{time}/checkpoint_{i}.pth'
    average_cost = test_checkpoint_model(checkpoint, test_dataset)
    writer.add_scalar('Metrics/Cost', average_cost, i)
    print(checkpoint)
    print(average_cost)

