
import pickle
import sys 
import networkx as nx
import torch
import numpy as np
from PG_structure_heuristic import State,update_state
from torch.utils.data import DataLoader
from collections import Counter
import random
def has_self_loop(graph, node):
    return node in graph.successors(node) 
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

def get_reward(tmplt_idx, cand_idx, state):
    g1 = state.g1
    g2 = state.g2

    # 计算模板图g1节点tmplt_idx的邻居集合
    neighbors_tmplt = set(g1[tmplt_idx])
    # 获取已匹配的模板图节点集合
    tmplt_matched_nodes = set(state.nn_mapping.keys())
    # 获取模板图邻居与已匹配节点的交集
    tmplt_node_intersection = neighbors_tmplt.intersection(tmplt_matched_nodes)
    # 获取候选图g2节点cand_idx对应于交集的节点
    cand_node_intersection = set([state.nn_mapping[n] for n in tmplt_node_intersection])
    # 计算候选图g2节点cand_idx的邻居集合
    neighbors_cand = set(g2[cand_idx])

    posi_reward = len(neighbors_cand.intersection(cand_node_intersection))
    nega_reward = len(tmplt_node_intersection) - posi_reward
    reward = posi_reward - nega_reward

    # 检查自环并奖励
    cycle_reward = 0
    if g1.has_edge(tmplt_idx, tmplt_idx) and g2.has_edge(cand_idx, cand_idx):
        cycle_reward = 1
    if g1.has_edge(tmplt_idx, tmplt_idx) and not g2.has_edge(cand_idx, cand_idx):
        cycle_reward = -1

    return reward + cycle_reward


def shuttle_node_id(G):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    # 创建一个映射，将原始节点映射到新的随机节点
    mapping = {original: new for original, new in zip(G.nodes(), nodes)}

    # 使用映射创建一个新的DiGraph
    H = nx.relabel_nodes(G, mapping)
    return H


def get_attr_dict(G):
    type_dict = {}

    # 遍历所有节点
    for node in G.nodes(data=True):
        node_name = node[0]
        node_type = node[1]['type']
        
        # 将节点名字添加到对应的类型列表中
        if node_type not in type_dict:
            type_dict[node_type] = set()
        type_dict[node_type].add(node_name)
    return type_dict


def calculate_cost(small_graph, big_graph, mapping):
    cost = 0
    for edge in small_graph.edges():
        # 根据映射找到大图中的对应节点
        mapped_edge = (mapping[edge[0]], mapping[edge[1]])
        
        # 检查对应的边是否在大图中
        if not big_graph.has_edge(*mapped_edge):
            cost += 1

    return cost



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

       self.g1.attr_dict =  get_attr_dict(self.g1)
       self.g2.attr_dict =  get_attr_dict(self.g2)
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
                          g1_reverse=state.g1_reverse,
                          g2_reverse=state.g2_reverse,
                          ori_candidates=state.ori_candidates)
        reward = get_reward(action[0],action[1],state)
        # update_state(new_state,self.threshold)
        if len(nn_mapping) == len(state.g1.nodes):
            return new_state,state,reward,True
        else:
            return new_state,state,reward,False




    
        
    
    
    