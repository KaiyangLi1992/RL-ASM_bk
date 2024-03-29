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


dataset = 'Email'
def random_walk_sample(graph, MAX_NODES_PER_SUBGRAPH):
    num_nodes_subgraph = MAX_NODES_PER_SUBGRAPH
    start_node = random.choice(list(graph.nodes))
    subgraph_nodes = set([start_node])
    current_node = start_node
    steps = 0
    while len(subgraph_nodes) < num_nodes_subgraph and steps < 1000:
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = random.choice(neighbors)
            subgraph_nodes.add(current_node)
        else:
            break  # Break if the node has no neighbors
        steps += 1  # Increment step count
    return graph.subgraph(subgraph_nodes)

def rename_id(G):
    nodes_list = list(G.nodes())
    node_id_mapping = {node: idx for idx, node in enumerate(nodes_list)}
    node_id_mapping_reverse = {idx: node for idx, node in enumerate(nodes_list)}
    G = nx.relabel_nodes(G, node_id_mapping, copy=True)
    return G,node_id_mapping_reverse 



def get_subgraphs(graph, MAX_NODES_PER_SUBGRAPH,dense,N):
    subgraphs = []
    if len(graph.nodes()) < 10:
        return None
    times = 0
    while len(subgraphs) < N:
        MAX_NODES_PER_SUBGRAPH = random.randint(10,20)
        sampled_subgraph = random_walk_sample(graph, MAX_NODES_PER_SUBGRAPH)
        avg_degree = sum(dict(sampled_subgraph.degree()).values()) / len(sampled_subgraph)
        # subgraphs.append(copy.deepcopy(sampled_subgraph))
        if dense:
            if avg_degree > 0:
                subgraphs.append(copy.deepcopy(sampled_subgraph))
                if len(subgraphs)%100 == 0:
                    print(f"Found {len(subgraphs)} subgraphs")
            times += 1
        if times > 100000:
            return subgraphs 
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

with open('./data/EMAIL/email_nx_graph.pkl','rb') as f:
    graph = pickle.load(f)

encoder, X = _get_enc_X(graph)
graphs = [graph]

RegularGraph_ls = []
for i in range(len(graphs)):
    graphs[i].graph['gid'] = i
    graphs[i],_ = rename_id(graphs[i])
    RegularGraph_ls.append(RegularGraph(graphs[i]))

for g in tqdm(graphs):
    subgraphs = get_subgraphs(g,8,True,10000)
    if subgraphs:
        for subgraph in subgraphs:
            sampled_subgraph = subgraph
            sampled_subgraph.graph['gid'] = len(RegularGraph_ls) 
            sampled_subgraph,matching = rename_id(sampled_subgraph)
            matchings[(sampled_subgraph.graph['gid'],g.graph['gid'])] = matching
            RegularGraph_ls.append(RegularGraph(sampled_subgraph))
            pairs[(sampled_subgraph.graph['gid'],g.graph['gid'])] = GraphPair()
            print( len(RegularGraph_ls) )
            


name = 'EMAIL'
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
with open(f'./data/EMAIL/EMAIL_dataset_sparse_n_10_20_num_01_31.pkl','wb') as f:
    pickle.dump(dataset_train,f)
with open(f'./data/EMAIL/EMAIL_dataset_sparse_n_10_20_num_01_31_matching.pkl','wb') as f:
    pickle.dump(matchings,f)











