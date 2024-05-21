
import pickle
import sys 
import networkx as nx
import torch
import numpy as np
from PG_structure_heuristic import State,update_state
from torch.utils.data import DataLoader
from collections import Counter
import random
import random
from environment import generate_ri_query_plan

class RandomSelector:
    def __init__(self, data):
        self.data = data
        self.shuffle_data()

    def shuffle_data(self):
        self.remaining = self.data[:]
        random.shuffle(self.remaining)

    def get_next_item(self):
        if not self.remaining:  # 如果列表为空，则重新洗牌
            self.shuffle_data()
        return self.remaining.pop()  # 选取并移除最后一个元素












class environment:
    def __init__(self,dataset):
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.searchtree = None
        self.g1 = None
        self.g2 = None
        self.threshold = np.inf
        pairs_list = list(dataset.pairs.keys())
        self.selector = RandomSelector(pairs_list)
        
      
        
    def reset(self):
       batch_gids = self.selector.get_next_item()
       self.g1 = self.dataset.look_up_graph_by_gid(batch_gids[0]).get_nxgraph()
       self.g2 = self.dataset.look_up_graph_by_gid(batch_gids[1]).get_nxgraph()

       state_init = State(self.g1,self.g2)
       self.threshold = np.inf
       self.order,_ = generate_ri_query_plan(self.g1)
       return state_init
    
    def step(self,state,action):
        nn_mapping = state.nn_mapping.copy()
        nn_mapping[action[0]] = action[1]
        state.pruned_space.append(action[1])
        new_state = State(state.g1,state.g2,
                          nn_mapping=nn_mapping,
                          candidates=state.candidates,
                          attr_sim=state.attr_sim)
        # reward = get_reward(action[0],action[1],state)
        # update_state(new_state,self.threshold)
        if len(nn_mapping) == len(state.g1.nodes):
            return new_state,state,True
        else:
            return new_state,state,False




    
        
    
    
    