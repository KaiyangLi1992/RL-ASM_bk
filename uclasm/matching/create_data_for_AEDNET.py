
import sys
import pickle
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/") 

with open('./data/SYNTHETIC/SYNTHETIC_testset_sparse_noiseratio_5.0_n_8_16_num_01_31.pkl','rb') as f:
    dataset = pickle.load(f)

graph_dict = {}

for pair in dataset.pairs.keys():
    g1 = dataset.look_up_graph_by_gid(pair[0]).get_nxgraph()
    g2 = dataset.look_up_graph_by_gid(pair[1]).get_nxgraph()
    graph_dict[pair[0]] = g1
    graph_dict[pair[1]] = g2


pairs = list(dataset.pairs.keys())


with open('./data/SYNTHETIC/for_AEDNET/SYNTHETIC_pairs_noiseratio_5.0.pkl','wb') as f:
    pickle.dump(pairs,f)

with open('./data/SYNTHETIC/for_AEDNET/SYNTHETIC_graph_dict_noiseratio_5.0.pkl','wb') as f:
    pickle.dump(graph_dict,f)

