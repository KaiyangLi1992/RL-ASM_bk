import networkx as nx
import sys 
from collections import defaultdict

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
import tqdm



def are_all_graphs_non_isomorphic(graph_list):
    n = len(graph_list)
    isomorphic_id = []
    for i in tqdm.tqdm(range(n)):
        # print(i)
        for j in range(i + 1, n):
            # print(graph_list[i].number_of_nodes(),graph_list[j].number_of_nodes())
            if nx.is_isomorphic(graph_list[i], graph_list[j]):
                isomorphic_id.append((i,j))
               
    return isomorphic_id

def check_isomorphic(dataset_name,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'
    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_noiseratio_{noiseratio*100}_n_{subgraph_node_num}_num_01_31_short.pkl','rb') as f:
        dataset = pickle.load(f)
    pairs = dataset.pairs.keys()

    grouped = defaultdict(list)
    for first, second in pairs:
        grouped[second].append(first)

    # 提取每个分组的第一个元素，并创建最终的列表
    subgraph_group = [group for group in grouped.values()]

    isomorphic_id_group = []
    for subgraph_list in subgraph_group:
        subgraph_ls = []
        for subgraph_id in subgraph_list:
            subgraph_ls.append(dataset.look_up_graph_by_gid(subgraph_id).get_nxgraph())
        isomorphic_id = are_all_graphs_non_isomorphic(subgraph_ls)
        isomorphic_id = [(subgraph_list[x], subgraph_list[y]) for x, y in isomorphic_id]
        isomorphic_id_group.append(isomorphic_id)

 


    rm_graphid = []
    remaining_nodes = []
    for i, sublist in enumerate(isomorphic_id_group):
        G = nx.Graph()
        G.add_edges_from(sublist)
        connected_components = list(nx.connected_components(G))
        for component in connected_components:
            sample = random.choice(list(component))
            remaining_nodes.extend([node for node in component if node != sample])
    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_noiseratio_{noiseratio*100}_n_{subgraph_node_num}_num_01_31_rm_graphid_short.pkl','wb') as f:
        pickle.dump(remaining_nodes,f)


if __name__ == '__main__':
    dataset_name = 'EMAIL'
    noiseratio = 0
    check_isomorphic(dataset_name,noiseratio)
    # dataset_name = 'EMAIL'
    # noiseratio = 0.05
    # check_isomorphic(dataset_name,noiseratio)
    # dataset_name = 'EMAIL'
    # noiseratio = 0.1
    # check_isomorphic(dataset_name,noiseratio)








    
    