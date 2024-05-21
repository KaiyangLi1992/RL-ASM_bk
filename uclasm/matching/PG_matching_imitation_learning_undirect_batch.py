import time
import torch.nn.functional as F
import pickle
import sys
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim

sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/uclasm/") 
sys.path.append("/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/GraphGPS/") 


from tqdm import tqdm
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import saver
from NSUBS.model.OurSGM.data_loader import get_data_loader_wrapper
from NSUBS.model.OurSGM.train import train
from NSUBS.model.OurSGM.test import test
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.src.utils import OurTimer, save_pickle
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.model.OurSGM.train import cross_entropy_smooth
from torch.utils.data import Dataset, DataLoader,ConcatDataset,Subset
from torch_geometric.data import Batch
from NSUBS.src.utils import from_networkx
from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li
from networkx.algorithms import isomorphism
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import networkx as nx
import random
import gc
import datetime
import torch.optim.lr_scheduler as lr_scheduler
# from PG_matching_ImitationLearning_concat import policy_network
from environment import environment, get_init_action, calculate_cost
import sys
import argparse
import torch_geometric.utils as pyg_utils
import os
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch, Data
import torch_geometric
torch_geometric.seed_everything(1)

def separate_graphs(batch_data):
    graphs = []
    pyg_li = batch_data.to_data_list()
    for single_graph in pyg_li:
        nx_graph = pyg_utils.to_networkx(single_graph, to_undirected=True, remove_self_loops=False)
        graphs.append(nx_graph)
    return graphs


device = torch.device(FLAGS.device)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
if FLAGS.dataset == 'AIDS':
    dim = 40
if FLAGS.dataset == 'SYNTHETIC':
    dim = 13
if FLAGS.dataset == 'EMAIL':
    dim = 47
if FLAGS.dataset == 'MSRC_21':
    dim = 27
# else:
#     dim = 47


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def _create_model(d_in_raw):
    if FLAGS.matching_order == 'nn':
        if FLAGS.load_model != 'None':
            load_replace_flags(FLAGS.load_model)
            saver.log_new_FLAGS_to_model_info()
            if FLAGS.glsearch:
                model = GLS() # create here since FLAGS have been updated_create_model
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC() # create here since FLAGS have been updated
            ld = torch.load(FLAGS.load_model, map_location=FLAGS.device)
            model.load_state_dict(ld)
            saver.log_info(f'Model loaded from {FLAGS.load_model}')
        else:
            if FLAGS.glsearch:
                model = GLS()
            else:
                model = create_dvn(d_in_raw, FLAGS.d_enc)
                # model = DGMC()
        saver.log_model_architecture(model, 'model')
        return model.to(FLAGS.device)
    else:
        return None
    



def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result




class Data_for_model_label(Dataset):
    def __init__(self, pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,ind_li,reward_li):
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li
        self.ind_li = ind_li
        self.reward_li = reward_li

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
        reward = self.reward_li[idx]

        return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind, reward


def my_collate_fn(batch):
    # batch is a list of tuples (dict_key, data_tensor)
    pyg_data_q_li = [item[0] for item in batch]
    pyg_data_t_li = [item[1] for item in batch]
    u = [item[2] for item in batch]
    v_li = [item[3] for item in batch]
    u2v_li_li = [item[4] for item in batch]
    nn_map_li = [item[5] for item in batch]
    ind_li = [item[6] for item in batch]
    reward_li = [item[7] for item in batch]

    for i in range(len(u2v_li_li)):
        u2v_li = u2v_li_li[i]
        pyg_data_t = pyg_data_t_li[i]
        assert max([max(x) for x in u2v_li.values()])  < pyg_data_t.num_nodes





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
            u2v_li[key + cumulative_nodes1] = [x + cumulative_nodes2 for x in value]
            assert max(value) < pyg_data_t_li[i].num_nodes
        cumulative_nodes1 += pyg_data_q_li[i].num_nodes
        cumulative_nodes2 += pyg_data_t_li[i].num_nodes
        


    return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind_li, reward_li 





dict_graph = {}
def _create_batch_data(pre_processed_li):
    global dict_graph
    pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,ind_li,reward_li = \
        [], [], [], [], [], [],[],[]
    for pre_processed in tqdm(pre_processed_li):
        g1, g2, u, v_li, nn_map, CS, candidate_map,ind,reward = pre_processed
        u2v_li = create_u2v_li(nn_map, CS, candidate_map)
        assert max([max(x) for x in u2v_li.values()]) < g2.number_of_nodes()


        if g1.graph['gid'] in dict_graph:
            pyg_data_q_li.append(dict_graph[g1.graph['gid']])
        else: 
            data1 = from_networkx(g1)
            # x = torch.concatenate((g1.init_x, g1.RWSE), dim=1)
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
            pyg_data_t_li.append(data2)
            dict_graph[g2.graph['gid']] = data2


        u_li.append(u)
        v_li_li.append(v_li)
        u2v_li_li.append(u2v_li)
        nn_map_li.append(nn_map)
        ind_li.append(ind)
        reward_li.append(reward)

    data_for_model_label = Data_for_model_label(pyg_data_q_li, pyg_data_t_li, u_li, v_li_li,
                                                 u2v_li_li, nn_map_li,ind_li,reward_li)
    
    # pickle.dump(data_for_model_label, open(f'data_for_model_label.pkl', 'wb'))
    # with open('./data/EMAIL/EMAIL_trainset_dense_noiseratio_all_packed_n_16_32_num_01_31_LapPE_imitationlearning_processed_li_RI_order.pkl', 'rb') as f:
    #     data_for_model_label = pickle.load(f)
    data_loader = DataLoader(data_for_model_label, batch_size=1024, collate_fn=my_collate_fn,shuffle=True)
    batch = next(iter(data_loader))
    return data_loader


def main():
    model = _create_model(dim).to(device)


    
    model.train()
    writer = SummaryWriter(f'plt_imitationlearning/{timestamp}')

    FLAGS.lr = 1e-3
    warmup_epochs = 500
    optimizer = optim.AdamW(model.parameters(), lr=FLAGS.lr,weight_decay=1e-3)
    lambda1 = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1
    


    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50000)


    checkpoint_path ='/home/kli16/ISM_custom/esm_NSUBS_RWSE_LapPE/esm_LapPE/ckpt_imitationlearning/2024-03-22_14-00-10/checkpoint_28000.pth'
    checkpoint = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 


    with open(f'./data/{FLAGS.dataset}/{FLAGS.dataset}_trainset_dense_noiseratio_0_n_16_32_num_01_31_LapPE_imitationlearning_processed_li_RI_order.pkl', 'rb') as f:
        pre_processed_li1 = pickle.load(f)
    with open(f'./data/{FLAGS.dataset}/{FLAGS.dataset}_trainset_dense_noiseratio_5.0_n_16_32_num_01_31_LapPE_imitationlearning_processed_li_RI_order.pkl', 'rb') as f:
        pre_processed_li2 = pickle.load(f)
    with open(f'./data/{FLAGS.dataset}/{FLAGS.dataset}_trainset_dense_noiseratio_10.0_n_16_32_num_01_31_LapPE_imitationlearning_processed_li_RI_order.pkl', 'rb') as f:
        pre_processed_li3 = pickle.load(f)



    pre_processed_li = pre_processed_li1 + pre_processed_li2+pre_processed_li3
    # # pre_processed_li1 = pre_processed_li1 * 120
    dataloader = _create_batch_data(pre_processed_li[:])
    loss_value = nn.MSELoss()

   



    step = 0
    for episode in range(1000):
        for batch in dataloader:
            if step < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler_cos.step()

            batch_input = batch[:-2]
            label = batch[-2]
            value_truth =  [0.68*i for i in batch[-1]]
          
            out_policy, out_value = \
                model(*batch_input,
                    True,
                    graph_filter=None, filter_key=None,
                )
            
            buffer_policy = []
           
            predicts = [] 
            for i,out in enumerate(out_policy):
                policy_truth = torch.zeros(out.shape,device=device)
                policy_truth[label[i]] = 1
                buffer_policy.append((out,policy_truth))
            
                max_index = torch.argmax(out)
                predicts.append(max_index.item())  

            loss_policy_batch = 0
            for out_policy,policy_truth in buffer_policy:
                ce_loss = cross_entropy_smooth(out_policy, policy_truth)
                loss_policy_batch += ce_loss

            loss_value_batch = loss_value(out_value,torch.tensor(value_truth).unsqueeze(1).to(device))

            loss_batch = loss_policy_batch + loss_value_batch
            
            ave_acc = sum(1 for x, y in zip(label, predicts) if x == y)/len(label)
            if step % 10 == 0:
                writer.add_scalar('Loss/total', loss_batch, step)
                writer.add_scalar('Loss/policy', loss_policy_batch, step)
                writer.add_scalar('Loss/value', loss_value_batch, step)
                writer.add_scalar('ACC', ave_acc, step)
                # writer.add_scalar('Value/0', out_value[0].item(), step)
                # writer.add_scalar('Value/3', out_value[3].item(), step)
                # writer.add_scalar('Value/7', out_value[7].item(), step)

                print(ave_acc)
                print(loss_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        
            


            if step % 2000 == 0:
            # 创建一个检查点每隔几个时期
                checkpoint = {
                    'epoch': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # ... (其他你想保存的元数据)
                }
                directory_name = f"ckpt_imitationlearning/{timestamp}/"
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)
                torch.save(checkpoint, f'ckpt_imitationlearning/{timestamp}/checkpoint_{step}.pth')
            step += 1
    

if __name__ == '__main__':
    main()



