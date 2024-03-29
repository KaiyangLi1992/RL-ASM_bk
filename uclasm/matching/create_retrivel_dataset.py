import networkx as nx
import sys 
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
from tqdm import tqdm
from collections import deque

dataset = 'EMAIL'
dense = True
num_min = 16
num_max = 32

def bfs_sampled_subgraph(G, n, m):
    """
    从图G中采样一个包含节点数在[n, m]区间内的导出子图。
    G: 原始图
    n: 最小节点数
    m: 最大节点数
    """


    if len(G.nodes()) < n:
        raise ValueError("图中的节点不足以形成所需大小的子图")
    while True:
        n_stop = random.randint(n, m)

        # 随机选择一个起始节点
        start_node = random.choice(list(G.nodes()))
        visited = {start_node}
        queue = deque([start_node])
        nodes_collected = [start_node]
        
        while queue and len(nodes_collected) < m:
            current_node = queue.popleft()
            for neighbor in G.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    nodes_collected.append(neighbor)
            if len(nodes_collected) >= n_stop:
                # 达到或超过n个节点时可以停止，根据需求调整
                # print(n)
                print(len(nodes_collected))
                break
        
        # 检查是否满足节点数要求
        if n <= len(nodes_collected) <= m:
            # 基于收集到的节点创建导出子图
            # print(len(nodes_collected))
            subgraph = G.subgraph(nodes_collected)
            return subgraph
      


def random_walk_sample(graph, MAX_NODES_PER_SUBGRAPH):
    num_nodes_subgraph = MAX_NODES_PER_SUBGRAPH
    start_node = random.choice(list(graph.nodes))
    subgraph_nodes = set([start_node])
    current_node = start_node
    edges_collected = set()
    while len(subgraph_nodes) < num_nodes_subgraph:
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            next_node = random.choice(neighbors)
            edge = tuple(sorted([current_node, next_node])) 
            edges_collected.add(edge)
            current_node = next_node
            subgraph_nodes.add(current_node)
        else:
            break  # Break if the node has no neighbors
    return  graph.edge_subgraph(edges_collected).copy()

def rename_id(G):
    nodes_list = list(G.nodes())
    node_id_mapping = {node: idx for idx, node in enumerate(nodes_list)}
    node_id_mapping_reverse = {idx: node for idx, node in enumerate(nodes_list)}
    G = nx.relabel_nodes(G, node_id_mapping, copy=True)
    return G,node_id_mapping_reverse 



def get_subgraphs(graph,dense,N):
    subgraphs = []
    if len(graph.nodes()) < 10:
        return None
    times = 0
   
    for i in tqdm(range(N)):  
        
        # avg_degree = sum(dict(sampled_subgraph.degree()).values()) / len(sampled_subgraph)
        # subgraphs.append(copy.deepcopy(sampled_subgraph))
        if not dense:
            MAX_NODES_PER_SUBGRAPH = random.randint(num_min,num_max)
            sampled_subgraph = random_walk_sample(graph, MAX_NODES_PER_SUBGRAPH)  
        else: 
            sampled_subgraph = bfs_sampled_subgraph(graph, num_min, num_max)

        subgraphs.append(copy.deepcopy(sampled_subgraph))
         
       
    return subgraphs


def read_graphs(adjacency_file, graph_indicator_file,node_label_file):
    # Step 1: Read graph indicators and build a mapping from node ID to graph ID
    with open(graph_indicator_file, 'r') as file:
        node_to_graph = {i+1: int(line.strip()) for i, line in enumerate(file)}

    # Identify the number of graphs
    max_graph_id = max(node_to_graph.values())
    graphs = [nx.Graph() for _ in range(max_graph_id)]

    with open(node_label_file, 'r') as file:
        node_labels = {i+1: int(line.strip()) for i, line in enumerate(file)}

    # Step 2: Read the adjacency file and add edges to the corresponding graphs
    with open(adjacency_file, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split(','))
            graph_id = node_to_graph[node1]
            if node1 not in graphs[graph_id - 1]:
                graphs[graph_id - 1].add_node(node1, type=node_labels[node1])
            if node2 not in graphs[graph_id - 1]:
                graphs[graph_id - 1].add_node(node2, type=node_labels[node2])


            graphs[graph_id - 1].add_edge(node1, node2)

    return graphs

# Usage
matchings = {}
pairs = {}
if dataset == 'EMAIL':
    with open('./data/EMAIL/email_nx_graph.pkl','rb') as f:
        graph = pickle.load(f)
        graphs = [graph]
else:
    graphs = read_graphs('./data/SYNTHETIC/SYNTHETIC_A.txt', './data/SYNTHETIC/SYNTHETIC_graph_indicator.txt','./data/SYNTHETIC/SYNTHETIC_node_labels.txt')





G_combined = nx.Graph()

    
for G in graphs:
    # 添加节点和边到新图中
    G_combined.add_nodes_from(G.nodes(data=True))
    G_combined.add_edges_from(G.edges(data=True))

encoder, X = _get_enc_X(G_combined)


RegularGraph_ls = []
for i in range(len(graphs)):
    graphs[i].graph['gid'] = i
    graphs[i],_ = rename_id(graphs[i])
    RegularGraph_ls.append(RegularGraph(graphs[i]))

for g in tqdm(graphs):
    subgraphs = get_subgraphs(g,dense,100)
    if subgraphs:
        for subgraph in subgraphs:
            sampled_subgraph = subgraph
            sampled_subgraph.graph['gid'] = len(RegularGraph_ls) 
            sampled_subgraph,matching = rename_id(sampled_subgraph)
            matchings[(sampled_subgraph.graph['gid'],g.graph['gid'])] = matching
            RegularGraph_ls.append(RegularGraph(sampled_subgraph))
            pairs[(sampled_subgraph.graph['gid'],g.graph['gid'])] = GraphPair()
            


name = dataset
natts = ['type']
eatts = [] 
tvt = 'train'
align_metric = 'sm'
node_ordering = 'bfs'
glabel = None

our_dataset = OurDataset(name, RegularGraph_ls, natts, eatts, pairs, tvt, align_metric, node_ordering, glabel, None)



for key,value in matchings.items():
    g1 = our_dataset.gs[key[0]].nxgraph
    g2 = our_dataset.gs[key[1]].nxgraph
    for u,v in value.items():
        assert(g1.nodes[u]['type'] == g2.nodes[v]['type'])




dataset_train, _ = encode_node_features_custom(our_dataset,encoder)
with open(f'./data/{dataset}/{dataset}_dataset_dense_n_{num_min}_{num_max}_num_01_31_short.pkl','wb') as f:
    pickle.dump(dataset_train,f)
with open(f'./data/{dataset}/{dataset}_dataset_dense_n_{num_min}_{num_max}_num_01_31_matching_short.pkl','wb') as f:
    pickle.dump(matchings,f)











