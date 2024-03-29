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
import time
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
from PG_matching_RL_undirect import _create_model,_get_CS,_preprocess_NSUBS,update_and_get_position
from PG_matching_RL_PPO import _create_batch_data,_preprocess_NSUBS

matching_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_testset_dense_n_16_num_200_01_16_matching.pkl'   # 获取文件名
with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)



def update_action_exp(state,action):
    gid = state.g1.graph['gid']
    matching = matchings[gid-1]
    action_1 = matching[action[0]]
    action = (action[0],action_1)
    return action

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
    
model = _create_model(47)


def test_checkpoint_model(ckpt_pth,test_dataset):
    with torch.no_grad():
        checkpoint = torch.load(ckpt_pth,map_location=torch.device(FLAGS.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        env = environment(test_dataset)
        action_preds = []
        action_labels = []
        for episode in range(50):
            state_init = env.reset()
            stack = [state_init]
            while stack:
                state = stack.pop()
                state.action_space = state.get_action_space(env.order)
                pre_processed = _preprocess_NSUBS(state,0)
                batch, data = _create_batch_data([pre_processed])
                batch=batch[:-1]
                model.eval()
                out_policy, out_value = \
                    model(*batch,
                        True,
                        graph_filter=None, filter_key=None,
                    )
                action_prob = F.softmax(out_policy[0] - out_policy[0].max()) + 1e-10
                _,action_ind = action_prob.max(0)
                

                action_pred = state.action_space[action_ind]
                action_preds.append(action_pred[1])
                action = update_action_exp(state,action_pred)
                action_labels.append(action[1])

                newstate, state,reward, done = env.step(state,action)
                stack.append(newstate)

                # predicts.append(max_index.item())   
                if done:
                    break
                
        # 返回总奖励，以便在后续可能使用
        count_diff = sum(1 for x, y in zip(action_preds, action_labels) if x == y)
        count_diff_0 = sum(1 for x, y in zip(action_preds[0::16], action_labels[0::16]) if x == y)
        count_diff_7 = sum(1 for x, y in zip(action_preds[7::16], action_labels[7::16]) if x == y)
        count_diff_15 = sum(1 for x, y in zip(action_preds[15::16], action_labels[15::16]) if x == y)
        return count_diff/len(action_preds),count_diff_0/len(action_preds[0::16]),count_diff_7/len(action_preds[0::16]),count_diff_15/len(action_preds[0::16])


# 使用该函数测试多个检查点
with open('/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_testset_dense_n_16_num_200_01_16_RWSE.pkl','rb') as f:
    test_dataset = pickle.load(f)


modelId = '2024-01-20_23-41-45'
# modelId = FLAGS.modelId
try:
    clear_directory(f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/runs_test_acc/{modelId}/')
except:
    pass
writer = SummaryWriter(f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/runs_test_acc/{modelId}/')
for i in range(70000, 100000, 2000):
    checkpoint = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/ckpt_imitationlearning/{modelId}/checkpoint_{i}.pth'
    average_acc,acc_0,acc_7,acc_15 = test_checkpoint_model(checkpoint, test_dataset)
    writer.add_scalar('Metrics/Cost', average_acc, i)
    print(checkpoint)
    print(average_acc)
    print(acc_0)
    print(acc_7)
    print(acc_15)

