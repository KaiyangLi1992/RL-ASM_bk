import networkx as nx
import sys 
from collections import defaultdict
import copy
sys.path.extend([
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm",
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/",
        "/home/kli16/esm_NSUBS_RWSE_LapPE/esm/NSUBS/",
    ])

from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair
import copy
import random
from node_feat import encode_node_features_custom,encode_node_features
import pickle
from NSUBS.model.OurSGM.data_loader import _get_enc_X
import networkx as nx




def add_noise_edges(graph, num_edges):
    """
    向图中添加num_edges条随机边
    """
    all_nodes = list(graph.nodes())
    for _ in range(num_edges):
        # 随机选择两个节点
        u, v = random.sample(all_nodes, 2)
        # 确保两个节点之间没有边
        while graph.has_edge(u, v):
            u, v = random.sample(all_nodes, 2)
        graph.add_edge(u, v)


def add_noise_nodes(G, num_nodes,labels):
    selected_nodes = random.sample(G.nodes(), num_nodes)
    for node in selected_nodes:
        current_type = G.nodes[node]['type']
        # 从 all_types 中去除当前节点的 'type'，以确保新的 'type' 与原来不同
        possible_types = labels - {current_type}
        # 随机选择一个新的 'type'
        new_type = random.choice(list(possible_types))
        # 更新节点属性
        G.nodes[node]['type'] = new_type


def collection_labels(dataset):
    graphs = dataset.gs
    labels = set()
    for graph in graphs:
        G = graph.get_nxgraph()
        for _, node_data in G.nodes(data=True):
            label = node_data.get('type', [])
            labels.add(label)
    return labels


def split_integer(n):
    a = random.randint(0, n)
    b = n - a
    return a, b


def  add_noise(dataset_name,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'

    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_n_{subgraph_node_num}_num_01_31_short.pkl','rb') as f:
        dataset = pickle.load(f)

    labels = collection_labels(dataset)
    for pairs in dataset.pairs.keys():
        g1 = dataset.look_up_graph_by_gid(pairs[0]).get_nxgraph()
        avg_degree = sum(dict(g1.degree()).values()) / len(g1)
    
        num_edges_nodes = round((g1.number_of_edges()+g1.number_of_nodes()) * noiseratio)
        num_edges,num_nodes = split_integer(num_edges_nodes)
        if num_nodes > g1.number_of_nodes():
            num_edges,num_nodes = num_nodes,num_edges
        add_noise_edges(g1, num_edges)
        add_noise_nodes(g1, num_nodes,labels)

    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_noiseratio_{noiseratio*100}_n_{subgraph_node_num}_num_01_31_short.pkl','wb') as f:
        pickle.dump(dataset,f)

if __name__ == '__main__':
    add_noise('EMAIL',0)

