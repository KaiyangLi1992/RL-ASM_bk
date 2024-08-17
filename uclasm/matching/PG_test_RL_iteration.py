import pickle
import sys 
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim
from sys_path_config import extend_sys_path
extend_sys_path()
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
from environment import calculate_cost
from config_loader import Config
dataset = FLAGS.dataset
Config.load_config(f"./config/{dataset}_PPO_config.yaml")
model = _create_model(FLAGS.dim)

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
        costs_wo_bt = []
        for episode in range(100): 
            print(f'episode:{episode}')
            start_time = time.time()
            duration = 600
            min_cost = np.inf
            state_init = env.reset()
            stack = [state_init]
            step = 0 
            state_init.candidates = state_init.get_candidates()
            cache_global_cost = {}
            cache_action_prob = set()
            while stack:
                step += 1 
                current_time = time.time()  # 获取当前时间
                if current_time - start_time > duration:
                    step = -1
                    break
                state = stack.pop()
                
                mapping_tuples = tuple(sorted(state.nn_mapping.items(), key=lambda item: item[0], reverse=True))
                if mapping_tuples not in cache_global_cost:
                    update_state(state,min_cost)
                    start_time_lower_bound = time.time()
                    cache_global_cost[mapping_tuples] = state.globalcosts
                    end_time_lower_bound = time.time()
                    times_lower_bound.append(end_time_lower_bound - start_time_lower_bound)
                else:
                    state.globalcosts = cache_global_cost[mapping_tuples]
                    
                
                
                if np.any(np.all(state.candidates == False, axis=1)):
                    continue
                state.action_space,terminal = state.get_action_space(env.order)
                if terminal:
                    continue

                if mapping_tuples not in cache_action_prob:
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

                    cache_action_prob.add(mapping_tuples)

                else:
                    _,action_ind = state.proba_cache.max(0)
                    action = state.action_space_cache[action_ind]


                
                state.proba_cache[action_ind] = -1
                new_state,state,  done = env.step(state, action)
                stack.append(state)   
            
                if done:
                    cost = calculate_cost(new_state.g1,new_state.g2,new_state.nn_mapping)
                    print(f"cost:{cost}")
                    if min_cost == np.inf:
                        costs_wo_bt.append(cost)
                    # print(np.min(new_state.get_globalcosts()))
                    if cost < min_cost:
                        min_cost = cost
                        cache_global_cost = {}
                        cache_action_prob = set()
                    if min_cost==0:
                        break
                    continue
                    # break

                else:
                    stack.append(new_state)
            # print(step)
            end_time = time.time() 
            steps.append(step)
            costs.append(min_cost)
            times.append(end_time - start_time)

        records = {}
        records['steps'] = steps
        records['costs'] = costs
        records['times'] = times
        records['times_lower_bound'] = times_lower_bound
        records['times_infer'] = times_infer
        records['costs_wo_bt'] = costs_wo_bt


    
        return records
if FLAGS.noiseratio == 0:
    noiseratio = '_noiseratio_0'
elif FLAGS.noiseratio == 5:
     noiseratio = '_noiseratio_5.0'
elif FLAGS.noiseratio == 10:
     noiseratio = '_noiseratio_10.0'


# with open('/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_testset_dense_noiseratio_2_n_16_num_200_01_16_RWSE.pkl','rb') as f:
with open(f'./data/{FLAGS.dataset}/{FLAGS.dataset}_testset_dense{noiseratio}_n_{FLAGS.subgraph_node_num}_num_01_31_LapPE.pkl','rb')  as f:
    test_dataset = pickle.load(f)
checkpoint = f'./ckpt_RL/MSRC_21_RL.pth'
records = test_checkpoint_model(checkpoint,test_dataset)

with open(f'records_{FLAGS.dataset}_noiseratio_{FLAGS.noiseratio}_RL_iteration.pkl','wb') as f:
    pickle.dump(records,f)

