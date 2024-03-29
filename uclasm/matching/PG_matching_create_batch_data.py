from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li
import torch
from NSUBS.src.utils import from_networkx
from torch.utils.data import Dataset, DataLoader, ConcatDataset
dict_graph = {}  
from torch_geometric.data import Data, Batch
from NSUBS.model.OurSGM.config import FLAGS
device = torch.device(FLAGS.device)

class Data_for_model_label(Dataset):
    def __init__(self, pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li):
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li
        self.ind_li = ind_li

    def __len__(self):
        return len(self.u_li)

    def __getitem__(self, idx):
        pyg_data_q = self.pyg_data_q_li[idx]
        pyg_data_t = self.pyg_data_t_li[idx]
        u = self.u_li[idx]
        v_li = self.v_li_li[idx]
        u2v_li = self.u2v_li_li[idx]
        nn_map = self.nn_map_li[idx]
        ind = self.ind_li[idx]

        return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind
    

def my_collate_fn(batch):
    pyg_data_q_li = [item[0] for item in batch]
    pyg_data_t_li = [item[1] for item in batch]
    u = [item[2] for item in batch]
    v_li = [item[3] for item in batch]
    u2v_li_li = [item[4] for item in batch]
    nn_map_li = [item[5] for item in batch]
    ind_li = [item[6] for item in batch]

    pyg_data_q = Batch.from_data_list(pyg_data_q_li).to(device)
    pyg_data_t = Batch.from_data_list(pyg_data_t_li).to(device)

    nn_map = {}
    cumulative_nodes1 = 0
    cumulative_nodes2 = 0
    for i, data in enumerate(pyg_data_q_li):
        nn_map_ori = nn_map_li[i]
        for key, value in nn_map_ori.items():
            nn_map[key + cumulative_nodes1] = value + cumulative_nodes2
        cumulative_nodes1 += pyg_data_q_li[i].num_nodes
        cumulative_nodes2 += pyg_data_t_li[i].num_nodes

    u2v_li = {}
    cumulative_nodes1 = 0
    cumulative_nodes2 = 0
    for i, data in enumerate(pyg_data_q_li):
        u2v_li_ori = u2v_li_li[i]
        for key, value in u2v_li_ori.items():
            u2v_li[key + cumulative_nodes1] = list({x + cumulative_nodes2 for x in value})
        cumulative_nodes1 += pyg_data_q_li[i].num_nodes
        cumulative_nodes2 += pyg_data_t_li[i].num_nodes

    ind = []
    cumulative_nodes2 = 0
    for i, data in enumerate(ind_li):
        ind.append(ind_li[i])
    # cumulative_nodes2 += pyg_data_t_li[i].num_nodes

    return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind
    
def create_batch_data(pre_processed_li, batch_size=1):
    global dict_graph
    pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li = [], [], [], [], [], [], []
    for pre_processed in pre_processed_li:
        g1, g2, u, v_li, nn_map, CS, candidate_map, ind = pre_processed
        u2v_li = create_u2v_li(nn_map, CS, candidate_map)
        assert max([max(x) for x in u2v_li.values()]) < g2.number_of_nodes()


        if g1.graph['gid'] in dict_graph:
            pyg_data_q_li.append(dict_graph[g1.graph['gid']])
        else: 
            data1 = from_networkx(g1)
            x = torch.concatenate((g1.init_x, g1.RWSE), dim=1)
            data1.x = x
            pyg_data_q_li.append(data1)
            dict_graph[g1.graph['gid']] = data1

        if g2.graph['gid'] in dict_graph:
            pyg_data_t_li.append(dict_graph[g2.graph['gid']])
        else: 
            data2 = from_networkx(g2)
            x = torch.concatenate((g2.init_x, g2.RWSE), dim=1)
            data2.x = x
            pyg_data_t_li.append(data2)
            dict_graph[g2.graph['gid']] = data2

        u_li.append(u)
        v_li_li.append(v_li)
        u2v_li_li.append(u2v_li)
        nn_map_li.append(nn_map)
        ind_li.append(ind)

    data_for_model_label = Data_for_model_label(pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,
                                                ind_li)
    data_loader = DataLoader(data_for_model_label, batch_size=batch_size, collate_fn=my_collate_fn)
    batch = next(iter(data_loader))
    return batch, data_for_model_label