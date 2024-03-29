import pickle
import os
import datetime
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)
import sys
sys.path.extend([
        "/home/kli16/esm_NSUBS_generaldistance/esm",
        "/home/kli16/esm_NSUBS_generaldistance/esm/uclasm/",
        "/home/kli16/esm_NSUBS_generaldistance/esm/GraphGPS/"
    ])

from graphgps.transform.posenc_stats import get_lap_decomp_stats
import pickle
import numpy as np
import sys
import time
from tqdm import tqdm
from yacs.config import CfgNode as CN
from torch_geometric.utils import (get_laplacian, 
                                   to_undirected, to_dense_adj, scatter)

from torch import Tensor
from NSUBS.src.utils import from_networkx



def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
    

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing



def create_feature_RWSE(dataset_name,tvt,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'
    with open(f'./data/{dataset_name}/{dataset_name}_{tvt}_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31_LapPE.pkl','rb') as f:
        dataset = pickle.load(f)
    for graph in dataset.gs:
        nx_graph = graph.nxgraph
        data = from_networkx(nx_graph)
        print(sorted(nx_graph[0]))

        zero_indices = (data.edge_index[0, :] == 0).nonzero(as_tuple=True)[0]

        # 获取第二行中这些索引处的元素
        selected_elements = data.edge_index[1, zero_indices]

        # 转换为列表并排序
        sorted_list = selected_elements.tolist()
        sorted_list.sort()

        print(sorted_list)


        skip_time = [i for  i in range(1,21) ]
        feature = get_rw_landing_probs(skip_time, data.edge_index)
        nx_graph.RWSE = feature


    with open(f'./data/{dataset_name}/{dataset_name}_{tvt}_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31_LapPE_RWSE.pkl','wb') as f:
        pickle.dump(dataset,f)



def create_feature_LapPE(dataset_name,tvt,noiseratio):
    if dataset_name == 'AIDS':
        subgraph_node_num = '6_10'
    if dataset_name == 'SYNTHETIC':
        subgraph_node_num = '16_32'
    if dataset_name == 'EMAIL':
        subgraph_node_num = '16_32'
    if dataset_name == 'MSRC_21':
        subgraph_node_num = '16_32'
    yaml_name = 'EMAIL_GateGCN_LapPE_RWSE.yaml'
    with open(yaml_name, 'r') as f:
        yaml_content = f.read()
    cfg = CN.load_cfg(yaml_content)
    with open(f'./data/{dataset_name}/{dataset_name}_{tvt}_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31.pkl','rb') as f:
        dataset = pickle.load(f)
    for graph in tqdm(dataset.gs):
        nx_graph = graph.nxgraph
        data = from_networkx(nx_graph)
        undir_edge_index = data.edge_index
        N = data.num_nodes

        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=None,num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())

        max_freqs=cfg.posenc_LapPE.eigen.max_freqs
        eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)
        
        nx_graph.EigVals = data.EigVals
        nx_graph.EigVecs = data.EigVecs 
    




    with open(f'./data/{dataset_name}/{dataset_name}_{tvt}_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31_LapPE.pkl','wb') as f:
        pickle.dump(dataset,f)


if __name__ == '__main__':
    dataset_name = 'EMAIL'
    noiseratio = 0
    create_feature_LapPE(dataset_name,'trainset',noiseratio)
    # create_feature_RWSE(dataset_name,'testset',noiseratio)
    # create_feature_RWSE(dataset_name,'validset',noiseratio)
    