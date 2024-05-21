from itertools import product
import random
import networkx as nx
import numpy as np
from laptools import clap
from matching_utils import MonotoneArray
from datetime import datetime
from NSUBS.model.OurSGM.config import FLAGS
from custom_clap import naive_costs

def update_state(state,threshold):
    state.threshold = threshold
    changed_cands = np.ones((len(state.g1.nodes),), dtype=np.bool_)
    state.candidates = state.get_candidates()
    # 从state_candidates中找出索引不在state_nn_mapping键中的行
    if np.any(np.all(state.candidates == False, axis=1)):
        return
    old_candidates = state.candidates.copy()
    # state.get_globalcosts()
    state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
    state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())

    # TODO: Does this break if nodewise changes the candidates?
    while True:
        state.candidates = state.get_candidates()
        if np.any(np.all(state.candidates == False, axis=1)):
            return
        state.localcosts[:] = np.maximum(state.localcosts, state.get_localcosts())
        state.globalcosts[:] = np.maximum(state.globalcosts, state.get_globalcosts())
        

        changed_cands = np.any(state.candidates != old_candidates, axis=1)
        if ~np.any(changed_cands):
            break
        old_candidates = state.candidates.copy()




cache = {}
def get_adjacency_matrix_with_cache(graph):
    # 使用图的某种唯一标识符作为缓存键
    graph_id = graph.graph['gid']
    
    if graph_id not in cache:
        cache[graph_id] = nx.adjacency_matrix(graph,sorted(graph.nodes()))
        
    return cache[graph_id]




def get_non_matching_mask(state):
        """Gets a boolean mask for the costs array corresponding to all entries
        that would violate the matching."""
        mask = np.zeros((len(state.g1.nodes),len(state.g2.nodes)), dtype=np.bool_)
        matching = [(k, v) for k, v in state.nn_mapping.items()]
        if len(matching) > 0:
            mask[[pair[0] for pair in matching],:] = True
            mask[:,[pair[1] for pair in matching]] = True
            mask[tuple(np.array(matching).T)] = False
        return mask

class State(object):
    def __init__(self,g1,g2,threshold=np.inf,pruned_space=None,\
                 candidates=None,nn_mapping={},attr_sim=None):
        self.g1 = g1
        self.g2 = g2
        self.hn = None
        self.cn = None
        self.nn_mapping = nn_mapping
        self.proba_cache = []
        self.action_space_cache = []

        if  attr_sim is None:
            self.attr_sim = self.get_attr_sim()
        else:
            self.attr_sim = attr_sim

       

        if candidates is None:
            self.candidates = self.generate_ori_candidate()
        else:
            self.candidates = candidates

        
        if pruned_space is None:
            self.pruned_space = []
        else:
            self.pruned_space = pruned_space
        self.shape = (len(g1.nodes),len(g2.nodes))

        self.globalcosts = np.zeros(self.shape).view(MonotoneArray)
        self.localcosts = np.zeros(self.shape).view(MonotoneArray)

        self.threshold = threshold
    def get_attr_sim(self):
        g1 = self.g1
        g2 = self.g2
        attrs_g1 = np.array([g1.nodes[i]['type'] for i in sorted(g1.nodes)])
        attrs_g2 = np.array([g2.nodes[i]['type'] for i in sorted(g2.nodes)])
        
        similarity_matrix = attrs_g1[:, None] == attrs_g2[None, :]
        
        return similarity_matrix

       
        
    

    def get_action_heuristic(self):
        row_index = len(self.nn_mapping.keys())
        row_candidates = self.candidates[row_index,:]

        true_candidates = np.where(row_candidates)[0]
        true_candidates_pruned = [node for node in true_candidates if node not in self.pruned_space]

        if not true_candidates_pruned: 
            return None,True


        row = self.globalcosts[row_index].copy()
        indices = np.arange(row.size)
        not_in_list = ~np.isin(indices, true_candidates_pruned)
        row[not_in_list] = np.inf
        

        min_col_index = np.argmin(row)
        min_index_tuple = (row_index,min_col_index)

        return min_index_tuple,False
    

    def get_action_space(self,order):

        matrix = self.candidates
        if FLAGS.order_Gq == 'random':
            row_index = len(self.nn_mapping)
        elif FLAGS.order_Gq == 'RI':
            row_index = order[len(self.nn_mapping)]
        action_u = row_index
        u_candidates_array = self.candidates[action_u,:]
        u_candidates_li = np.where(u_candidates_array)[0]
        u_candidates_li = [node for node in u_candidates_li if node not in self.pruned_space]

        if not u_candidates_li: 
            return None,True
     
        # 转换为 (x, y) 格式的坐标
        action_space = [(action_u, node) for node in u_candidates_li]
        
        return action_space,False

    
        
  
    
    
    def get_candidates(self):
        candidates =  (self.globalcosts < (self.threshold - 1e-8)).view(np.ndarray)
        return candidates

    
        
    def generate_ori_candidate(self):
        similarity_matrix = np.ones((self.g1.number_of_nodes(),self.g2.number_of_nodes(),),dtype=np.bool_) 
        return similarity_matrix
    
    
    def get_localcosts(self):
        g1 = self.g1
        g2 = self.g2
        candidates = self.candidates
        local_costs = np.zeros((len(g1.nodes),len(g2.nodes)))

        for dst_idx, src_idx in list(nx.Graph(g1).edges()):
            
            src_is_cand = candidates[src_idx]
            dst_is_cand = candidates[dst_idx]
            supported_edges = None

        
            total_tmplt_edges = 0
            tmplt_adj = get_adjacency_matrix_with_cache(g1)
            world_adj = get_adjacency_matrix_with_cache(g2)
            tmplt_adj_val = tmplt_adj[src_idx, dst_idx]
            total_tmplt_edges += tmplt_adj_val

            # if there are no edges in this channel of the template, skip it
            if tmplt_adj_val == 0:
                continue

            # sub adjacency matrix corresponding to edges from the source
            # candidates to the destination candidates
            world_sub_adj = world_adj[:, dst_is_cand][src_is_cand, :]

            # Edges are supported up to the number of edges in the template
            if supported_edges is None:
                supported_edges = world_sub_adj.minimum(tmplt_adj_val)
            else:
                supported_edges += world_sub_adj.minimum(tmplt_adj_val)

            src_support = supported_edges.max(axis=1)
            src_least_cost = total_tmplt_edges - src_support.A

    

            src_least_cost = np.array(src_least_cost).flatten()
            # Update the local cost bound
            local_costs[src_idx][src_is_cand] += src_least_cost

            # if src_idx != dst_idx:
            dst_support = supported_edges.max(axis=0)
            dst_least_cost = total_tmplt_edges - dst_support.A
            dst_least_cost = np.array(dst_least_cost).flatten()
            local_costs[dst_idx][dst_is_cand] += dst_least_cost
        local_costs += ~self.attr_sim
        return local_costs



        
    def get_globalcosts(self):
        g1 = self.g1
        g2 = self.g2
        tmplt_idx_mask = np.ones(len(g1.nodes), dtype=np.bool_)
        world_idx_mask = np.ones(len(g2.nodes), dtype=np.bool_)


        global_costs = np.zeros((len(g1.nodes),len(g2.nodes)))
        for tmplt_idx, world_idx in self.nn_mapping.items():
            tmplt_idx_mask[tmplt_idx] = False
            world_idx_mask[world_idx] = False
        local_costs = self.localcosts
        local_costs[~self.candidates] = 100000
        partial_match_cost = np.sum([local_costs[match]/2  for match in self.nn_mapping.items()])
        mask = np.ix_(tmplt_idx_mask, world_idx_mask)
        total_match_cost = partial_match_cost
        if np.any(tmplt_idx_mask):
            costs = local_costs[mask] / 2 
            # global_cost_bounds = naive_costs(costs)
            global_cost_bounds = clap.costs(costs)
            # assert np.array_equal(global_cost_bounds, global_cost_bounds)
            global_costs[mask] = np.maximum( global_costs[mask],global_cost_bounds) + partial_match_cost
            total_match_cost += np.min(global_cost_bounds)
        non_matching_mask = get_non_matching_mask(self)
        global_costs[non_matching_mask] = float("inf")
        for tmplt_idx, world_idx in self.nn_mapping.items():
            global_costs[tmplt_idx, world_idx] = np.maximum(global_costs[tmplt_idx, world_idx],total_match_cost)
        return global_costs


