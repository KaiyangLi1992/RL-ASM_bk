import os
import datetime
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import pickle
import numpy as np
import sys
import time
from torch.nn.utils import  weight_norm




sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/") 
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from NSUBS.src.utils import from_networkx
from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li
# Custom module imports
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import ParameterSaver
from NSUBS.model.OurSGM.train import train, cross_entropy_smooth
from NSUBS.model.OurSGM.model_glsearch import GLS
from NSUBS.model.OurSGM.utils_our import load_replace_flags
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from NSUBS.src.utils import OurTimer, save_pickle
from environment import environment,  calculate_cost

from torch.optim.lr_scheduler import StepLR
import torch_geometric
torch_geometric.seed_everything(1)

dataset_file_name = '/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/SYNTHETIC/SYNTHETIC_trainset_dense_noiseratio_0_5_10_n_16_32_num_01_31_RWSE.pkl'
# matching_file_name = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_debug/esm/data/unEmail_trainset_dense_n_16_num_10000_01_16_matching.pkl'
if FLAGS.dataset == 'AIDS':
    dim = 40
if FLAGS.dataset == 'SYNTHETIC':
    dim = 13
if FLAGS.dataset == 'EMAIL':
    dim = 47


timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = torch.device(FLAGS.device)
imitationlearning = True


# checkpoint_path = '/home/kli16/ISM_custom/esm_NSUBS_RWSE_trans_batch/esm/ckpt_RL/2024-01-06_11-31-00/checkpoint_9500.pth'
# checkpoint_path = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/ckpt_imitationlearning/2024-02-20_22-35-37/checkpoint_48000.pth'
# checkpoint = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))
# lr_decay = True
# T_max = int(5e4)
# lr = FLAGS.lr
# weight_decay = 0
# epochs = 1e6

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
    
dict_graph = {}  
def _create_batch_data(pre_processed_li, batch_size=1):
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
            data1.x = g1.init_x
            data1.EigVals = g1.EigVals
            data1.EigVecs = g1.EigVecs
            pyg_data_q_li.append(data1)
            dict_graph[g1.graph['gid']] = data1

        if g2.graph['gid'] in dict_graph:
            pyg_data_t_li.append(dict_graph[g2.graph['gid']])
        else: 
            data2 = from_networkx(g2)
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

    data_for_model_label = Data_for_model_label(pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li,
                                                ind_li)
    data_loader = DataLoader(data_for_model_label, batch_size=batch_size, collate_fn=my_collate_fn)
    batch = next(iter(data_loader))
    return batch, data_for_model_label

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




def _create_model(d_in_raw):
    model = create_dvn(d_in_raw, FLAGS.d_enc)
    # saver.log_model_architecture(model, 'model')
    return model.to(FLAGS.device)

    


# with open(dataset_file_name,'rb') as f:
#     dataset = pickle.load(f)

# with open(matching_file_name,'rb') as f:
#     matchings = pickle.load(f)


def _get_CS(state,g1,g2):
    result = {i: np.where(row)[0].tolist() for i, row in enumerate(state.candidates)}
    return result


def _preprocess_NSUBS(state, ind):
    g1 = state.g1
    g2 = state.g2
    u = state.action_space[0][0]
    v_li = [action[1] for action in state.action_space]
    CS = _get_CS(state, g1, g2)
    nn_map = state.nn_mapping
    candidate_map = {u: v_li}
    return (g1, g2, u, v_li, nn_map, CS, candidate_map, ind)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr = lr
        self.buffer = RolloutBuffer()

        self.policy = _create_model(dim).to(device)
   
        if imitationlearning is True:
            self.policy.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = optim.AdamW(self.policy.parameters(), self.lr,weight_decay=weight_decay)
        if imitationlearning is True:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        if lr_decay:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)

        self.policy_old = _create_model(dim).to(device)


        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()



    def select_action(self, state,env):
            with torch.no_grad():
                state.action_space = state.get_action_space(env.order)
                pre_processed = _preprocess_NSUBS(state,0)
                batch, data = _create_batch_data([pre_processed])
                batch=batch[:-1]
                self.policy_old.eval()
                out_policy, out_value,  = \
                    self.policy_old(*batch,
                        True,
                        graph_filter=None, filter_key=None,
                )
                action_prob = F.softmax(out_policy[0] - out_policy[0].max()) + 1e-10
                m = Categorical(action_prob)
                action_ind = m.sample()
                action = state.action_space[action_ind]
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(m.log_prob(action_ind))
            self.buffer.state_values.append(out_value[0])

            return action

    def policy_evaluate(self,states,actions,is_terminals):
        action_logprobs_li = []
        dist_entropy_li= []
        state_values_li = []

        pre_processed_li = []
        for s in states:
            pre_processed = _preprocess_NSUBS(s, 0)
            pre_processed_li.append(pre_processed)

        batch, data = _create_batch_data(pre_processed_li,len(states))
        batch = batch[:-1]
         ######### train or eval 
        self.policy.train()
        out_policy, out_value = \
                self.policy(*batch,
                            True,
                            graph_filter=None, filter_key=None,
                            )
        
        out_policy_li_ = out_policy
        out_value_li_ = out_value
        for out_policy, out_value, action, state in zip(out_policy_li_, out_value_li_, actions, states):
            action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
            ind = state.action_space.index(action)
            dist = Categorical(action_prob)
            action_logprob = dist.log_prob(torch.tensor(ind).to(device))
            # dist_entropy = dist.entropy()
            state_value = out_value

            action_logprobs_li.append(action_logprob)
            dist_entropy_li.append(dist.entropy())
            state_values_li.append(state_value)
       
        return torch.stack(action_logprobs_li), torch.stack(state_values_li), torch.stack(dist_entropy_li)




    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            start_time =   time.time() 
            
            logprobs, state_values, dist_entropy = self.policy_evaluate(old_states, old_actions,self.buffer.is_terminals)
            

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss_policy = -torch.min(surr1, surr2)
            loss_value = 0.5 * self.MseLoss(state_values, rewards)
            loss_dist_entropy = - 0.01 * dist_entropy
            loss = loss_policy + loss_value + loss_dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()

            loss.mean().backward()
            self.optimizer.step()
            end_time = time.time() 
            execution_time = end_time - start_time  # 计算执行时间
            # print(f"程序执行时间: {execution_time} 秒")
            # loss.mean().item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return loss.mean().item(),loss_policy.mean().item(),loss_value.mean().item(),loss_dist_entropy.mean().item()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

################ PPO hyperparameters ################


# update_timestep = 1 * 640   # update policy every n timesteps
# K_epochs = 3        # update policy for K epochs
# eps_clip = 0.1             # clip parameter for PPO
# gamma = 0.99         # discount factor
# max_training_episodes = int(1e6)

# random_seed = 0         # set random seed if required (0 = no random seed)

# saverP = ParameterSaver()
# if imitationlearning is True:
#     checkpoint_path_para = checkpoint_path
# else:
#     checkpoint_path_para = None
# if lr_decay is True:
#     T_max_para = T_max
# else:
#     T_max_para = None

# parameters = {
#     'pid': os.getpid(),
#     'file_path':os.path.abspath(__file__),
#     'learning_rate': lr,
#     'checkpoint_path': checkpoint_path_para,
#     'T_max' : T_max_para,
#     'gamma':gamma,
#     'weight_decay':weight_decay

# }
# saverP.save(parameters,file_name=f'{timestamp}.log')

# ###################### logging ######################
# #### log files for multiple runs are NOT overwritten

# log_dir = "PPO_logs"
# if not os.path.exists(log_dir):
#       os.makedirs(log_dir)

# log_dir = log_dir + '/' + timestamp + '/'
# if not os.path.exists(log_dir):
#       os.makedirs(log_dir)


# #### create new log file for each run 
# log_f_name = log_dir + '/PPO_' + timestamp + "_log.txt"

# print("current logging run number for " + timestamp)
# print("logging at : " + log_f_name)

# ############# print all hyperparameters #############
# log_f = open(log_f_name,"w+")
# log_f.write("--------------------------------------------------------------------------------------------\n")


# log_f.write("PPO update frequency : " + str(update_timestep) + " timesteps\n") 
# log_f.write(f"PPO K epochs : {K_epochs}\n")
# log_f.write(f"PPO epsilon clip : {eps_clip}\n")
# log_f.write(f"discount factor (gamma) : {gamma}\n", )
# log_f.write(f"learning rate : {lr}\n")
# log_f.write("--------------------------------------------------------------------------------------------\n")
# log_f.write(f"d_enc : {FLAGS.d_enc}\n")
# log_f.write(f"device : {FLAGS.device}\n")
# if imitationlearning is True:
#     log_f.write(f"im_ckpt_path : {checkpoint_path}\n")

# log_f.write("--------------------------------------------------------------------------------------------\n")




def main():
    # model = _create_model(dim).to(device)
    
    # model.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter(f'plt_RL/{timestamp}')
    env  = environment(dataset)
    ppo_agent = PPO(lr, gamma, K_epochs, eps_clip)
   

    for episode in range(max_training_episodes):

        state_init = env.reset()
        # update_state(state_init,env.threshold)
        stack = [state_init]    
           
        while stack:
            state = stack.pop()
            action = ppo_agent.select_action(state,env)
            newstate, state,reward, done = env.step(state,action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            stack.append(newstate)
            if done:
                cost = calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping)
                ppo_agent.policy.reset_cache()
                break
       

        if episode % update_timestep == 0:
            
            loss,loss_policy,loss_value,loss_dist_entropy = ppo_agent.update()
            
            
        if lr_decay:
            ppo_agent.scheduler.step()
         


        
        if episode % 10 == 0:
            writer.add_scalar('Cost', cost, episode)
            writer.add_scalar('Loss/total', loss, episode)
            writer.add_scalar('Loss/policy', loss_policy, episode)
            writer.add_scalar('Loss/value', loss_value, episode)
            writer.add_scalar('Loss/dist_entropy', loss_dist_entropy, episode)
            print(f"Cost:{cost}")
            print(f"Lost:{loss}")


        if episode % 500 == 0:
        # 创建一个检查点每隔几个时期
            checkpoint = {
                'epoch': episode,
                'model_state_dict': ppo_agent.policy_old.state_dict(),
                'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                # ... (其他你想保存的元数据)
            }
            directory_name = f"ckpt_RL/{timestamp}/"
            if not os.path.exists(directory_name):
                os.makedirs(directory_name)
            torch.save(checkpoint, f'ckpt_RL/{timestamp}/checkpoint_{episode}.pth')
    

if __name__ == '__main__':
    main()
    log_f.close()



