dataset_name = 'EMAIL'
noiseratio = 0
tvt = 'trainset'
if dataset_name == 'EMAIL':
    subgraph_node_num = '16_32'
import sys
sys.path.extend([
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_trans_batch/esm/uclasm/",
        "/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm/NSUBS/",
    ])

yaml_name = 'EMAIL_GateGCN_LapPE_RWSE.yaml'
from yacs.config import CfgNode as CN
with open(yaml_name, 'r') as f:
    yaml_content = f.read()
cfg = CN.load_cfg(yaml_content)
import pickle
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from NSUBS.src.utils import from_networkx
from torch_geometric.graphgym.config import cfg
with open(f'./data/{dataset_name}/{dataset_name}_{tvt}_dense_noiseratio_{100*noiseratio}_n_{subgraph_node_num}_num_01_31_LapPE_short.pkl','rb') as f:
    dataset = pickle.load(f)


node_encoder = LapPENodeEncoder(cfg.gnn.dim_inner)
g = dataset.look_up_graph_by_gid(0).get_nxgraph()
data = from_networkx(g)
data.EigVals = g.EigVals
data.EigVecs = g.EigVecs

feats = node_encoder(data)

print(feats.shape)



