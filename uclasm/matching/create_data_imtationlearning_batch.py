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

sys.path.extend([
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_trans_batch/esm/uclasm/",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm/NSUBS/",
    ])


from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth
from torch.utils.data import Dataset, DataLoader,ConcatDataset,Subset
from torch_geometric.data import Batch
# from torch_geometric.utils import from_networkx
from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
# from PG_structure import update_state
import random
import gc
import datetime
import torch.optim.lr_scheduler as lr_scheduler
# from PG_matching_ImitationLearning_concat import policy_network
from environment import environment, get_init_action, calculate_cost
import sys
import argparse

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
device = torch.device(FLAGS.device)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

dim = 13

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def _create_model(d_in_raw):
    if FLAGS.matching_order == 'nn':
        if FLAGS.load_model != 'None':
            load_replace_flags(FLAGS.load_model)
            saver.log_new_FLAGS_to_model_info()
            if FLAGS.glsearch:
                model = GLS() # create here since FLAGS have been updated_create_model
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC() # create here since FLAGS have been updated
            ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
            model.load_state_dict(ld)
            saver.log_info(f'Model loaded from {FLAGS.load_model}')
        else:
            if FLAGS.glsearch:
                model = GLS()
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC()
        saver.log_model_architecture(model, 'model')
        return model.to(FLAGS.device)
    else:
        return None
    

def update_and_get_position(lst, tup):
    assert tup in lst
    return lst.index(tup),lst  # 返回tuple在list中的位置
    
    




def update_action_exp(state,action,matchings):
    gid1 = state.g1.graph['gid']
    gid2 = state.g2.graph['gid']
    matching = matchings[(gid1,gid2)]
    action_1 = matching[action[0]]
    action = (action[0],action_1)
    return action



def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result

class Data_for_model(Dataset):
    def __init__(self, pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li):
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li

    def __len__(self):
        return len(self.u_li)

    def __getitem__(self, idx):
        pyg_data_q = self.pyg_data_q_li[idx]
        pyg_data_t = self.pyg_data_t_li[idx]
        u = self.u_li[idx]
        v_li = self.v_li_li[idx]
        u2v_li = self.u2v_li_li[idx]
        nn_map = self.nn_map_li[idx]

        return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map
    


class Data_for_model_label(Dataset):
    def __init__(self, pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,ind_li):
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li
        self.ind_li = ind_li

    def __len__(self):
        return len(self.u_li)

    def __getitem__(self, idx):
        pyg_data_q = self.pyg_data_q_li[idx]
        pyg_data_t = self.pyg_data_t_li[idx]
        u = self.u_li[idx]
        v_li = self.v_li_li[idx]
        u2v_li = self.u2v_li_li[idx]
        nn_map = self.nn_map_li[idx]
        ind = self.ind_li[idx]

        return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map,ind


def my_collate_fn(batch):
    # batch is a list of tuples (dict_key, data_tensor)
    pyg_data_q_li = [item[0] for item in batch]
    pyg_data_t_li = [item[1] for item in batch]
    u = [item[2] for item in batch]
    v_li = [item[3] for item in batch]
    u2v_li_li = [item[4] for item in batch]
    nn_map_li = [item[5] for item in batch]
    ind_li = [item[6] for item in batch]

    pyg_data_q = Batch.from_data_list(pyg_data_q_li).to(device)
    pyg_data_t = Batch.from_data_list(pyg_data_t_li).to(device)

    nn_map = {}
    cumulative_nodes1 = 0
    cumulative_nodes2 = 0
    for i, data in enumerate(pyg_data_q_li):
        nn_map_ori = nn_map_li[i]
        for key, value in nn_map_ori.items():
            nn_map[key + cumulative_nodes1] = value + cumulative_nodes2
        cumulative_nodes1 += pyg_data_q_li[i].num_nodes
        cumulative_nodes2 += pyg_data_t_li[i].num_nodes

    u2v_li = {}
    cumulative_nodes1 = 0
    cumulative_nodes2 = 0
    for i, data in enumerate(pyg_data_q_li):
        u2v_li_ori = u2v_li_li[i]
        for key, value in u2v_li_ori.items():
            u2v_li[key + cumulative_nodes1] = list({x + cumulative_nodes2 for x in value})
        cumulative_nodes1 += pyg_data_q_li[i].num_nodes
        cumulative_nodes2 += pyg_data_t_li[i].num_nodes


    ind = []
    cumulative_nodes2 = 0
    for i, data in enumerate(ind_li):
        ind.append(ind_li[i])


    return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind



def _preprocess_NSUBS(state):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state,g1,g2)
    nn_map = state.nn_mapping
    candidate_map = {u:v_li}
    assert max(max(candidate_map.values())) < g2.number_of_nodes()
    return (g1,g2,u,v_li,nn_map,CS,candidate_map)




def create_batch(dataset_name,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'
    if dataset_name == 'MSRC_21':
        subgraph_node_num = '16_32'

    dataset_file_name = f'./data/{dataset_name}/{dataset_name}_trainset_dense_noiseratio_{noiseratio*100}_n_{subgraph_node_num}_num_01_31_LapPE.pkl'   # 获取文件名
    matching_file_name = f'./data/{dataset_name}/{dataset_name}_dataset_dense_n_{subgraph_node_num}_num_01_31_matching.pkl'   # 获取文件名
    with open(dataset_file_name,'rb') as f:
        dataset = pickle.load(f)

    with open(matching_file_name,'rb') as f:
        matchings = pickle.load(f)
    # device = torch.device(FLAGS.device)
    # print(f"Using device: {device}")
    # model = _create_model(dim).to(device)
    # model.load_state_dict(checkpoint_loaded['model_state_dict'])
    # writer = SummaryWriter(f'plt_imitationlearning/{timestamp}')

    env  = environment(dataset)
 
    rewards = []
    pre_processed_all = []

    for episode in range(len(dataset.pairs.keys())):
        pre_processed_ep = []
        rewards = []
        state_init = env.reset_create_data(episode)
        stack = [state_init]
        labels = []
        is_terminals =[] 
        while stack:
            state = stack.pop()
            state.action_space = state.get_action_space(env.order)
            action_exp = update_action_exp(state,state.action_space[0],matchings)
            ind,state.action_space = update_and_get_position(state.action_space, action_exp)
            pre_processed = _preprocess_NSUBS(state)


            newstate, state,reward, done = env.step(state,action_exp)
            rewards.append(reward)
            stack.append(newstate)

            labels.append(ind) 
            pre_processed_ep.append([*pre_processed, ind, 0])
            is_terminals.append(False)
            if done:
                is_terminals.append(True)
                # model.reset_cache()
                break
           
        accu_rewards = []        
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (0.99 * discounted_reward)
            accu_rewards.insert(0, discounted_reward)

        for i,accu_reward in enumerate(accu_rewards):
            pre_processed_ep[i][-1] = accu_reward
        pre_processed_all.extend(pre_processed_ep)

    with open(f'./data/{dataset_name}/{dataset_name}_trainset_dense_noiseratio_{noiseratio*100}_n_{subgraph_node_num}_num_01_31_LapPE_imitationlearning_processed_li_RI_order.pkl','wb') as f:
        pickle.dump(pre_processed_all,f)


        

def main():
    create_batch('EMAIL',0)     
    

if __name__ == '__main__':
    main()
