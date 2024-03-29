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


def add_dataset(dataset1,dataset2,match1,match2):
    RegularGraph_ls1 = dataset1.gs
    max_gid = max([g.get_nxgraph().graph['gid'] for g  in RegularGraph_ls1])+1

    RegularGraph_ls1 = dataset1.gs
    pairs1 = dataset1.pairs

    RegularGraph_ls2 = dataset2.gs
    for g in RegularGraph_ls2:
        g.get_nxgraph().graph['gid'] += max_gid


    pairs2 = dataset2.pairs
    pairs2_new = {}
    for key,value in pairs2.items():
        key_new = (key[0]+max_gid , key[1]+max_gid )
        pairs2_new[key_new] = value

    RegularGraph_ls1 += RegularGraph_ls2
    pairs1.update(pairs2_new)

    gid_ls = [g.get_nxgraph().graph['gid']  for g in RegularGraph_ls1]



    name = 'MSRC_21'
    natts = ['type']
    eatts = [] 
    tvt = 'train'
    align_metric = 'sm'
    node_ordering = 'bfs'
    glabel = None

    our_dataset = OurDataset(name, RegularGraph_ls1, natts, eatts, pairs1, tvt, align_metric, node_ordering, glabel, None)



    
    match2_new = {}
    for key,value in match2.items():
        key_new = (key[0]+max_gid , key[1]+max_gid )
        match2_new[key_new] = value

    match1.update(match2_new)
    return our_dataset,match1



if __name__ == '__main__':
    dataset_file_name = '/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/MSRC_21/MSRC_21_trainset_dense_noiseratio_0_n_16_32_num_01_31_LapPE.pkl'   # 获取文件名
    with open(dataset_file_name,'rb') as f:
        dataset1 = pickle.load(f)
    dataset_file_name = '/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/MSRC_21/MSRC_21_trainset_dense_noiseratio_5.0_n_16_32_num_01_31_LapPE.pkl'   # 获取文件名
    with open(dataset_file_name,'rb') as f:
        dataset2 = pickle.load(f)
    dataset_file_name = '/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/MSRC_21/MSRC_21_trainset_dense_noiseratio_10.0_n_16_32_num_01_31_LapPE.pkl'   # 获取文件名
    with open(dataset_file_name,'rb') as f:
        dataset3 = pickle.load(f)
    match_file_name = '/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/MSRC_21/MSRC_21_dataset_dense_n_16_32_num_01_31_matching.pkl'   # 获取文件名
    with open(match_file_name,'rb') as f:
        match1 = pickle.load(f)


    dataset,match = add_dataset(dataset1,dataset2,match1,match1)
    dataset,match = add_dataset(dataset,dataset3,match,match1)


with open(f'./data/MSRC_21/MSRC_21_trainset_dense_noiseratio_0_5_10_n_16_32_num_01_31_LapPE.pkl','wb') as f:
    pickle.dump(dataset,f)

match_file_name = './data/MSRC_21/MSRC_21_dataset_dense_noiseratio_0_5_10_n_16_32_num_01_31_matching.pkl'   # 获取文件名
with open(match_file_name,'wb') as f:
    pickle.dump(match,f)


