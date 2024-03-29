
import pickle
import sys 
import networkx as nx
import torch
# sys.path.append("/home/kli16/ISM_custom/esm/") 
# sys.path.append("/home/kli16/ISM_custom/esm/rlmodel") 
# sys.path.append("/home/kli16/ISM_custom/esm/uclasm/") 
import numpy as np
# from search.data_structures_search_tree import SearchTree
from PG_structure import State
from torch.utils.data import DataLoader
from collections import Counter
import random
def has_self_loop(graph, node):
    return node in graph.successors(node) 


import random

def compute_candidates(set1, set2):
    return len(set(set1) & set(set2))

# Function to generate RI Query Plan
def generate_ri_query_plan(query_graph):
    query_vertices_num = query_graph.number_of_nodes()
    order = [None] * query_vertices_num
    pivot = [None] * query_vertices_num

    visited = [False] * query_vertices_num
    # Select the vertex with the maximum degree as the start vertex.
    order[0] = max(query_graph.nodes, key=query_graph.degree)
    visited[order[0]] = True

    # Order vertices.
    for i in range(1, query_vertices_num):
        max_bn = -1
        tie_vertices = []
        for u in query_graph.nodes:
            if not visited[u]:
                cur_bn = sum(1 for v in query_graph.neighbors(u) if visited[v])
                if cur_bn > max_bn:
                    max_bn = cur_bn
                    tie_vertices = [u]
                elif cur_bn == max_bn:
                    tie_vertices.append(u)

        if len(tie_vertices) > 1:
            count = -1
            for u in tie_vertices:
                u_neighbors = set(query_graph.neighbors(u))
                cur_count = sum(1 for v in order[:i] if v is not None and compute_candidates(u_neighbors, set(query_graph.neighbors(v))) > 0)
                if cur_count > count:
                    count = cur_count
                    tie_vertices = [u]
                elif cur_count == count:
                    tie_vertices.append(u)

        if len(tie_vertices) > 1:
            count = -1
            for u in tie_vertices:
                u_neighbors = set(query_graph.neighbors(u))
                cur_count = sum(1 for v in u_neighbors if not visited[v] and all(not query_graph.has_edge(v, w) for w in order[:i] if w is not None))
                if cur_count > count:
                    count = cur_count
                    tie_vertices = [u]
                elif cur_count == count:
                    tie_vertices.append(u)

        order[i] = tie_vertices[0]
        visited[order[i]] = True
        for j in range(i):
            if order[j] is not None and query_graph.has_edge(order[i], order[j]):
                pivot[i] = order[j]
                break

    return order, pivot


class RandomSelector:
    def __init__(self, data):
        self.data = data
        self.shuffle_data()

    def shuffle_data(self):
        self.remaining = self.data[:]
        random.shuffle(self.remaining)
        self.remaining = self.remaining
        
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
    reward = posi_reward  - nega_reward

    # 检查自环并奖励
    cycle_reward = 0
    if g1.has_edge(tmplt_idx, tmplt_idx) and g2.has_edge(cand_idx, cand_idx):
        cycle_reward = 1
    if g1.has_edge(tmplt_idx, tmplt_idx) and not g2.has_edge(cand_idx, cand_idx):
        cycle_reward = -1

    if g1.nodes[tmplt_idx]['type'] == g2.nodes[cand_idx]['type']:
        node_reward = 1
    else:
        node_reward = -1


    return reward + cycle_reward + node_reward

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




def get_init_action(coordinates,globalcost):
        filtered_coordinates = [coord for coord in coordinates if coord != (-1, -1)]

        # 获取坐标及其对应的值
        coord_values = [(coord, globalcost[coord]) for coord in filtered_coordinates]

        # 根据值排序并取第一个坐标
        min_coord = min(coord_values, key=lambda x: x[1])[0]

        return min_coord

def calculate_cost(small_graph, big_graph, mapping):
    cost = 0
    for key,value in mapping.items():
        if small_graph.nodes[key]['type'] != big_graph.nodes[value]['type']:
            cost += 1

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
        self.order = None
        # self.gen = get_next_item(dataset)


    def reset_create_data(self,episode):
        batch_gids = self.selector.get_next_item()
        # batch_gids = ('31045','30304')

        self.g1 = self.dataset.look_up_graph_by_gid(batch_gids[0]).get_nxgraph()
        self.g2 = self.dataset.look_up_graph_by_gid(batch_gids[1]).get_nxgraph()

        #    self_loops = [(u, v) for u, v in self.g1.edges() if u == v]A
        #    self.g1.remove_edges_from(self_loops)


        #    self.g1 = shuttle_node_id(self.g1)


        self.g1.attr_dict =  get_attr_dict(self.g1)
        self.g2.attr_dict =  get_attr_dict(self.g2)
        state_init = State(self.g1,self.g2)
        self.threshold = np.inf
        self.order,_ = generate_ri_query_plan(self.g1)

        return state_init
        
    def reset(self):
       batch_gids = self.selector.get_next_item()
    #    batch_gids = [torch.tensor([1]), torch.tensor([0])]
       self.g1 = self.dataset.look_up_graph_by_gid(batch_gids[0]).get_nxgraph()
    #    self.g1 = self.dataset.look_up_graph_by_gid(episode+1).get_nxgraph()
       self.g2 = self.dataset.look_up_graph_by_gid(batch_gids[1]).get_nxgraph()

    #    self_loops = [(u, v) for u, v in self.g1.edges() if u == v]
    #    self.g1.remove_edges_from(self_loops)


    #    self.g1 = shuttle_node_id(self.g1)


       self.g1.attr_dict =  get_attr_dict(self.g1)
       self.g2.attr_dict =  get_attr_dict(self.g2)
       state_init = State(self.g1,self.g2)
       self.threshold = np.inf
       self.order,_ = generate_ri_query_plan(self.g1)
       
       return state_init
    
    def step(self,state,action):
        nn_mapping = state.nn_mapping.copy()
        nn_mapping[action[0]] = action[1]
        # state.pruned_space.append((action[0],action[1]))
        state.ori_candidates[:,action[1]] = False
        state.ori_candidates[action[0],:] = False
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




    
        

if __name__ == '__main__':
    with open('Ourdataset_toy_dataset.pkl','rb') as f:
        dataset = pickle.load(f)
    env  = environment(dataset)
    state_init = env.reset()
    action_space = state_init.get_action_space()
    init_action = get_init_action(action_space)
    new_state,_,done = env.step(state_init,init_action)


    
    
    