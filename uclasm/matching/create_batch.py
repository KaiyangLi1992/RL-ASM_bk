from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import torch
from NSUBS.src.utils import from_networkx
from NSUBS.model.OurSGM.config import FLAGS 
import numpy as np
device = FLAGS.device 
def create_u2v_li(nn_map, cs_map, candidate_map):
    u2v_li = {}
    for u in cs_map.keys():
        if u in nn_map:
            v_li = [nn_map[u]]
        elif u in candidate_map:
            v_li = candidate_map[u]
        else:
            v_li = cs_map[u]
        u2v_li[u] = v_li
    return u2v_li

def _preprocess_NSUBS(state, ind):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state, g1, g2)
    nn_map = state.nn_mapping
    candidate_map = {u: v_li}
    return (g1, g2, u, v_li, nn_map, CS, candidate_map, ind)

dict_graph = {}  
def create_batch_data(pre_processed_li, batch_size=1,old_data=None,include_label=False):
    global dict_graph
    if include_label:
        old_logprobs,advantages,rewards,old_states,old_actions = old_data
        old_logprobs_li = old_logprobs
        advantages_li = advantages
        rewards_li = rewards
        old_states_li = old_states
        old_actions_li = old_actions
    
        old_data_li = (old_logprobs_li,advantages_li,rewards_li,old_states_li,old_actions_li)

    pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li = [], [], [], [], [], [], []
    

    for pre_processed in pre_processed_li:
        g1, g2, u, v_li, nn_map, CS, candidate_map, ind = pre_processed
        u2v_li = create_u2v_li(nn_map, CS, candidate_map)
        assert max([max(x) for x in u2v_li.values()]) < g2.number_of_nodes()


        if g1.graph['gid'] in dict_graph:
            pyg_data_q_li.append(dict_graph[g1.graph['gid']])
        else: 
            data1 = from_networkx(g1)
            # x = torch.concatenate((g1.init_x, g1.RWSE), dim=1)
            # data1.x = x
            data1.x = g1.init_x
            data1.EigVals = g1.EigVals
            data1.EigVecs = g1.EigVecs
            pyg_data_q_li.append(data1)
            dict_graph[g1.graph['gid']] = data1

        if g2.graph['gid'] in dict_graph:
            pyg_data_t_li.append(dict_graph[g2.graph['gid']])
        else: 
            data2 = from_networkx(g2)
            # x = torch.concatenate((g2.init_x, g2.RWSE), dim=1)
            data2.x = g2.init_x
            data2.EigVals = g2.EigVals
            data2.EigVecs = g2.EigVecs
            # data2.x = x
            pyg_data_t_li.append(data2)
            dict_graph[g2.graph['gid']] = data2

        u_li.append(u)
        v_li_li.append(v_li)
        u2v_li_li.append(u2v_li)
        nn_map_li.append(nn_map)
        ind_li.append(ind)

        data_li = (pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,ind_li)
        
    if include_label:
        data_for_model_label = Data_for_model(data_li,old_data_li,include_label=True)
        data_loader = DataLoader(data_for_model_label, batch_size=batch_size, collate_fn=create_collate_fn(include_label=True))
        batch = next(iter(data_loader))
        return batch, data_loader
    else:
        data_for_model = Data_for_model(data_li,None,include_label=False)
        data_loader = DataLoader(data_for_model, batch_size=batch_size, collate_fn=create_collate_fn())
        batch = next(iter(data_loader))
        return batch, data_loader


def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result


def create_collate_fn(include_label=False):
    def my_collate_fn(batch):
        pyg_data_q_li = [item[0] for item in batch]
        pyg_data_t_li = [item[1] for item in batch]
        u = [item[2] for item in batch]
        v_li = [item[3] for item in batch]
        u2v_li_li = [item[4] for item in batch]
        nn_map_li = [item[5] for item in batch]
        ind_li = [item[6] for item in batch]
        if include_label:
            old_logprobs_li = [item[7] for item in batch]
            advantages_li = [item[8] for item in batch]
            rewards_li = [item[9] for item in batch]
            old_states_li = [item[10] for item in batch]
            old_actions_li =  [item[11] for item in batch]

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
        if not include_label:
            return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind
        else:

            old_logprobs = []
            for i, data in enumerate(old_logprobs_li):
                old_logprobs.append(old_logprobs_li[i])
            advantages = []
            for i, data in enumerate(advantages_li):
                advantages.append(advantages_li[i])

            rewards = []
            for i, data in enumerate(rewards_li):
                rewards.append(rewards_li[i])

            old_states = []
            for i, data in enumerate(old_states_li):
                old_states.append(old_states_li[i])

            old_actions = []
            for i, data in enumerate(old_actions_li):
                old_actions.append(old_actions_li[i])
        

            return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind, old_logprobs, advantages,rewards,old_states,old_actions
    return my_collate_fn

class Data_for_model(Dataset):
    def __init__(self, data_li,old_data_li=None,include_label=False):
        self.include_label = include_label

        pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li = data_li
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li
        self.ind_li = ind_li

        if self.include_label:
            old_logprobs_li,advantages_li,rewards_li,old_states_li,old_actions_li = old_data_li
            self.old_logprobs_li = old_logprobs_li
            self.advantages_li = advantages_li
            self.rewards_li = rewards_li
            self.old_states_li = old_states_li
            self.old_actions_li = old_actions_li

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
        if self.include_label is True:
            old_logprobs = self.old_logprobs_li[idx]
            advantages = self.advantages_li[idx]
            rewards = self.rewards_li[idx]
            old_states = self.old_states_li[idx]
            old_actions = self.old_actions_li[idx]

            return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind,old_logprobs,advantages,rewards,old_states,old_actions
        else:
            return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind