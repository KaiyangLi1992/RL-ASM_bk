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

def deduplication_dataset(dataset_name,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'
    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31_rm_graphid.pkl','rb') as f:
        deduplication_id = pickle.load(f)

    with open(f'./data/{dataset_name}/{dataset_name}_dataset_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31.pkl','rb') as f:
        dataset = pickle.load(f)

    pairs =  dataset.pairs.keys()

    deduplicated_pair = []
    for pair in pairs:
        if pair[0] not in deduplication_id and pair[1] not in deduplication_id:
            deduplicated_pair.append(pair)

    random.shuffle(deduplicated_pair)

    # 然后，根据8:1:1的比例计算分割点
    total_length = len(deduplicated_pair)
    split1 = int(total_length * 0.8)  # 第一段占80%
    split2 = split1 + int(total_length * 0.1)  # 第二段再占10%

    # 分割列表
    trainset_pairs_id = deduplicated_pair[:split1]
    validset_pairs_id = deduplicated_pair[split1:split2]
    testset_pairs_id = deduplicated_pair[split2:]

    trainset_pairs,validset_pairs,testset_pairs = {},{},{}

    for key,value in dataset.pairs.items():
        if key in trainset_pairs_id:
            trainset_pairs[key] = value
        if key in validset_pairs_id:
            validset_pairs[key] = value
        if key in testset_pairs_id:
            testset_pairs[key] = value


    dataset.pairs = trainset_pairs
    with open(f'./data/{dataset_name}/{dataset_name}_trainset_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31.pkl','wb') as f:
        pickle.dump(dataset,f)
    dataset.pairs = validset_pairs
    with open(f'./data/{dataset_name}/{dataset_name}_validset_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31.pkl','wb') as f:
        pickle.dump(dataset,f)
    dataset.pairs = testset_pairs
    with open(f'./data/{dataset_name}/{dataset_name}_testset_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31.pkl','wb') as f:
        pickle.dump(dataset,f)

if __name__ == '__main__':
    dataset_name = 'EMAIL'
    noiseratio = 0
    deduplication_dataset(dataset_name,noiseratio)
    # dataset_name = 'EMAIL'
    # noiseratio = 0.05
    # deduplication_dataset(dataset_name,noiseratio)
    # dataset_name = 'EMAIL'
    # noiseratio = 0.1
    # deduplication_dataset(dataset_name,noiseratio)








