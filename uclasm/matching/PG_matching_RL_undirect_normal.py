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
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/uclasm/") 


from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver,ParameterSaver
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth




import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
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



from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver,ParameterSaver
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth




import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
# from PG_structure import update_state
import random
import gc
import datetime

# from PG_matching_ImitationLearning_concat import policy_network
from environment import environment,get_init_action,calculate_cost
import sys
import argparse
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

dataset_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/HPRD_trainset_dense_n_16_num_10000_12_04_RWSE.pkl'   # 获取文件名
matching_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/HPRD_trainset_dense_n_16_num_10000_12_04_matching.pkl'   # 获取文件名
dim = 312
gamma = 0.99
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = torch.device(FLAGS.device)
checkpoint_path = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_reorder/esm/ckpt_imitationlearning/2023-12-04_15-32-09/checkpoint_30000.pth'

T_max = 2e4
lr = 1e-3

def _create_model(d_in_raw):
    model = create_dvn(d_in_raw, FLAGS.d_enc)
    saver.log_model_architecture(model, 'model')
    return model.to(device)

with open(dataset_file_name,'rb') as f:
    dataset = pickle.load(f)

with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)

def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result

def _preprocess_NSUBS(state):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state,g1,g2)
    nn_map = state.nn_mapping
    candidate_map = {u:v_li}
    return (g1,g2,u,v_li,nn_map,CS,candidate_map)




def main():

    model = _create_model(dim).to(device)
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter(f'plt_RL/{timestamp}')
    env  = environment(dataset)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for episode in range(50000):
        state_init = env.reset()
        stack = [state_init]    
        rewards = []
        saved_log_probs = []
        next_u = 0
        while stack:
            state = stack.pop()
            state.action_space = state.get_action_space(env.order)
            pre_processed = _preprocess_NSUBS(state)
            out_policy, out_value, out_other = \
                    model(*pre_processed,
                        True,
                        graph_filter=None, filter_key=None,
                )
            action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
            m = Categorical(action_prob)
            action_ind = m.sample()
            action = state.action_space[action_ind]
            newstate, state,reward, done = env.step(state,action)
            rewards.append(reward)
            stack.append(newstate)
            saved_log_probs.append(m.log_prob(action_ind))

            # predicts.append(max_index.item())   
            if done:
                cost = calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping)
                break

        
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        loss = []
        for log_prob, R in zip(saved_log_probs, returns):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

       
        writer.add_scalar('Loss', loss, episode)
        writer.add_scalar('Cost', cost, episode)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Loss', loss, episode)
        if episode%10==0:
            print(f"episode: {episode} loss: {loss}")
            print(f"episode: {episode} cost: {cost}")


        if episode % 1000 == 0:
        # 创建一个检查点每隔几个时期
            checkpoint = {
                'epoch': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # ... (其他你想保存的元数据)
            }
            directory_name = f"ckpt_RL/{timestamp}/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            torch.save(checkpoint, f'ckpt_RL/{timestamp}/checkpoint_{episode}.pth')
    

if __name__ == '__main__':
    main()



