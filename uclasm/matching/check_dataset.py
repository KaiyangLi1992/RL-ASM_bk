
import pickle
import networkx as nx
from networkx.algorithms import isomorphism
import sys
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_batch/esm/")
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_batch/esm/uclasm/")
from graph import OurGraph, RegularGraph, BioGraph
dataset_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_trainset_dense_n_8_num_10000_12_23_RWSE.pkl'   # 获取文件名
matching_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_trainset_dense_n_8_num_10000_12_23_matching.pkl'   # 获取文件名


with open(dataset_file_name,'rb') as f:
    dataset = pickle.load(f)

with open(matching_file_name,'rb') as f:
    matchings = pickle.load(f)


for i in range(0,10000):
    g1 = dataset.look_up_graph_by_gid(i+1).get_nxgraph()

    g2 = dataset.look_up_graph_by_gid(0).get_nxgraph()


    matching = matchings[i]

    u_li = matching.keys()
    v_li = matching.values()

    induced_subgraph1 = g1.subgraph(u_li)
    induced_subgraph2 = g2.subgraph(v_li)

    # 创建一个匹配器对象
    matcher = isomorphism.GraphMatcher(induced_subgraph1, induced_subgraph2)

    # 判断两个图是否同构
    isomorphic = matcher.is_isomorphic()
    assert isomorphic


import torch_geometric.utils as pyg_utils
import os
from torch_geometric.data import Batch, Data


with open('./data/unEmail_trainset_dens_0.2_n_8_num_10000_12_23_RWSE_imitationlearning_processed_li.pkl', 'rb') as f:
# with open('./data/unEmail_trainset_dens_0.2_n_8_num_2000_10_05_RWSE_imitationlearning_processed_li.pkl', 'rb') as f:
    pre_processed_li = pickle.load(f)

for pre_processed in pre_processed_li:
    pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind, reward = pre_processed


def separate_graphs(batch_data):
    graphs = []
    for graph_idx in batch_data.batch.unique():
        mask = batch_data.batch == graph_idx
        node_features = batch_data.x[mask]
        edge_index = batch_data.edge_index[:, mask]
        # 如果有其他节点或边的特征，也可以在这里提取
        single_graph = Data(x=node_features, edge_index=edge_index)
        nx_graph = pyg_utils.to_networkx(single_graph, to_undirected=True, remove_self_loops=False)
        graphs.append(nx_graph)
    return graphs




    
