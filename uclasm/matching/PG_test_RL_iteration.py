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
from matching.environment_heuristic import environment
from NSUBS.model.OurSGM.config import FLAGS
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from matching.PG_structure_heuristic import update_state
import torch_geometric
from PG_matching_RL_PPO import _create_batch_data,_preprocess_NSUBS,_create_model
import time
torch_geometric.seed_everything(1)
import copy

def calculate_cost(small_graph, big_graph, mapping):
    cost = 0
    for edge in small_graph.edges():
        mapped_edge = (mapping[edge[0]], mapping[edge[1]])
        if not big_graph.has_edge(*mapped_edge):
            cost += 1
    return cost

model = _create_model(40)

def test_checkpoint_model(ckpt_pth,test_dataset):
    
    # 加载测试环境
    with torch.no_grad():
        env = environment(test_dataset)
        checkpoint = torch.load(ckpt_pth,map_location=torch.device(FLAGS.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        steps = []
        costs = []
        times = []
        times_lower_bound = []
        times_infer = []
        for episode in range(len(test_dataset.pairs.keys())): 
            start_time = time.time()
            min_cost = np.inf
            state_init = env.reset()
            stack = [state_init]
            step = 0 
            state_init.candidates = state_init.get_candidates()
            cache_mapping = set()
            while stack:
                step += 1
                if step > 5000:
                    step = -1
                    break
                state = stack.pop()
                start_time_lower_bound = time.time()
                mapping_tuples = tuple(sorted(state.nn_mapping.items(), key=lambda item: item[0], reverse=True))
                if mapping_tuples not in cache_mapping:
                    update_state(state,min_cost)
                    
                end_time_lower_bound = time.time()
                times_lower_bound.append(end_time_lower_bound - start_time_lower_bound)
                if np.any(np.all(state.candidates == False, axis=1)):
                    continue
                state.action_space,terminal = state.get_action_space(env.order)
                if terminal:
                    continue

                if mapping_tuples not in cache_mapping:
                    start_time_infer = time.time()
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

                    state.action_space_cache = copy.deepcopy(state.action_space)
                    state.proba_cache = copy.deepcopy(action_prob)
                    end_time_infer = time.time()
                    times_infer.append(end_time_infer - start_time_infer)

                    cache_mapping.add(mapping_tuples)

                else:
                    _,action_ind = state.proba_cache.max(0)
                    action = state.action_space_cache[action_ind]


                
                state.proba_cache[action_ind] = -1
                new_state,state, reward, done = env.step(state, action)
                stack.append(state)   
            
                if done:
                    cost = calculate_cost(new_state.g1,new_state.g2,new_state.nn_mapping)
                    # print(np.min(new_state.get_globalcosts()))
                    if cost < min_cost:
                        min_cost = cost
                        cache_mapping = set()
                    if min_cost==0:
                        break
                    continue

                else:
                    stack.append(new_state)
            # print(step)
            end_time = time.time() 
            steps.append(step)
            costs.append(cost)
            times.append(end_time - start_time)

        records = {}
        records['steps'] = steps
        records['costs'] = min_cost
        records['times'] = times
        records['times_lower_bound'] = times_lower_bound
        records['times_infer'] = times_infer


    
        return records


# 使用该函数测试多个检查点
# with open('/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_testset_dense_noiseratio_2_n_16_num_200_01_16_RWSE.pkl','rb') as f:
with open('./data/AIDS/AIDS_testset_sparse_n_6_10_num_01_31_RWSE.pkl','rb')  as f:
    test_dataset = pickle.load(f)
checkpoint = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/ckpt_imitationlearning/2024-02-03_12-18-49/checkpoint_50000.pth'
records = test_checkpoint_model(checkpoint,test_dataset)

with open('records_2024-02-03_12-18-49_noiseratio_0_whole_matching.pkl','wb') as f:
    pickle.dump(records,f)

