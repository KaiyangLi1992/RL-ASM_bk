import os
import datetime
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
import copy
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/") 
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from NSUBS.src.utils import from_networkx
from NSUBS.model.OurSGM.dvn_wrapper import create_u2v_li
# Custom module imports
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import ParameterSaver
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from environment import environment,  calculate_cost,get_reward
from PG_matching_create_batch_data import create_batch_data
from PG_structure import State
from buffer import RolloutBuffer
import torch_geometric
import torch.multiprocessing as mp
import multiprocessing
import psutil
import threading
import queue
mp.set_start_method('spawn', force=True)
torch_geometric.seed_everything(1)

dataset_file_name = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/{FLAGS.dataset}/{FLAGS.dataset}_trainset_{FLAGS.dataset_type}_noiseratio_0_5_10_n_16_32_num_01_31_RWSE.pkl'
# dataset_file_name = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/data/SYNTHETIC/SYNTHETIC_trainset_dense_noiseratio_0_n_16_32_num_01_31_RWSE.pkl'
with open(dataset_file_name,'rb') as f:
    dataset = pickle.load(f)

dim = FLAGS.dim
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
device = torch.device(FLAGS.device)

# Hyperparameters
imitationlearning = True
lr_decay = True
checkpoint_path = f'/home/kli16/esm_NSUBS_RWSE_LapPE/esm/ckpt_imitationlearning/2024-02-20_22-35-37/checkpoint_48000.pth'
checkpoint = torch.load(checkpoint_path,map_location=torch.device(FLAGS.device))
T_max = int(5e4)
lr = FLAGS.lr
weight_decay = 0
epochs = 1e6
def print_open_files_count():
    pid = os.getpid()
    current_process = psutil.Process(pid)
    files = current_process.open_files()
    print(f"当前打开的文件数量: {len(files)}")

class Data_for_model_label(Dataset):
    def __init__(self, pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li,old_logprobs_li,
                 advantages_li,rewards_li,old_states_li,old_actions_li):
        self.pyg_data_q_li = pyg_data_q_li
        self.pyg_data_t_li = pyg_data_t_li
        self.u_li = u_li
        self.v_li_li = v_li_li
        self.u2v_li_li = u2v_li_li
        self.nn_map_li = nn_map_li
        self.ind_li = ind_li
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
        old_logprobs = self.old_logprobs_li[idx]
        advantages = self.advantages_li[idx]
        rewards = self.rewards_li[idx]
        old_states = self.old_states_li[idx]
        old_actions = self.old_actions_li[idx]

        return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind,old_logprobs,advantages,rewards,old_states,old_actions
    
dict_graph = {}  
def _create_batch_data_parallel(pre_processed_li, old_logprobs,advantages,rewards,old_states,old_actions,batch_size=1):
    global dict_graph
    pyg_data_q_li, pyg_data_t_li, u_li, v_li_li, u2v_li_li, nn_map_li, ind_li = [], [], [], [], [], [], []
    old_logprobs_li = old_logprobs
    advantages_li = advantages
    rewards_li = rewards
    old_states_li = old_states
    old_actions_li = old_actions

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
                                                ind_li,old_logprobs_li,advantages_li,rewards_li,old_states_li,old_actions_li)
    data_loader = DataLoader(data_for_model_label, batch_size=batch_size, collate_fn=my_collate_fn)
    batch = next(iter(data_loader))
    return batch, data_loader

def my_collate_fn(batch):
    pyg_data_q_li = [item[0] for item in batch]
    pyg_data_t_li = [item[1] for item in batch]
    u = [item[2] for item in batch]
    v_li = [item[3] for item in batch]
    u2v_li_li = [item[4] for item in batch]
    nn_map_li = [item[5] for item in batch]
    ind_li = [item[6] for item in batch]
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

    old_logprobs = []
    for i, data in enumerate(old_logprobs_li):
        old_logprobs.append(old_logprobs_li[i])
    advantages = []
    for i, data in enumerate(advantages_li):
        advantages.append(advantages_li[i])

    rewards = []
    for i, data in enumerate(rewards_li):
        rewards.append(rewards_li[i])
    # cumulative_nodes2 += pyg_data_t_li[i].num_nodes
    old_states = []
    for i, data in enumerate(old_states_li):
        old_states.append(old_states_li[i])

    old_actions = []
    for i, data in enumerate(old_actions_li):
        old_actions.append(old_actions_li[i])
   

    return pyg_data_q, pyg_data_t, u, v_li, u2v_li, nn_map, ind, old_logprobs, advantages,rewards,old_states,old_actions

def _create_model(d_in_raw):
    model = create_dvn(d_in_raw, FLAGS.d_enc)
    # saver.log_model_architecture(model, 'model')
    return model.to(FLAGS.device)

    
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






class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip,batch_size):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lr = lr
        self.batch_size = batch_size
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
        old_logprobs = torch.squeeze(torch.tensor(self.buffer.logprobs)).detach().to(device)
        old_state_values = torch.squeeze(torch.tensor(self.buffer.state_values)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        pre_processed_li = []
        for s in old_states:
            pre_processed = _preprocess_NSUBS(s, 0)
            pre_processed_li.append(pre_processed)

        
        batch, dataloader = _create_batch_data_parallel(pre_processed_li,old_logprobs,advantages,rewards,old_states,old_actions,self.batch_size)
        
        if len(old_states)//self.batch_size == 0:
            iter_num = int(len(old_states)//self.batch_size)
        else:
            iter_num = int(len(old_states)//self.batch_size)+1
    

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            start_time =   time.time() 
            loss_policy_li,loss_value_li,loss_dist_entropy_li = [],[],[] 
            # logprobs, state_values, dist_entropy = self.policy_evaluate(old_states, old_actions,self.buffer.is_terminals)
            for step, batch in enumerate(dataloader):
                action_logprobs_li = []
                dist_entropy_li= []
                state_values_li = []
                old_logprobs = batch[-5]
                advantages =  batch[-4]
                rewards = batch[-3]
                old_states = batch[-2]
                old_actions = batch[-1]
                batch = batch[:-6]
                ######### train or eval 
                self.policy.train()
                out_policy, out_value = \
                        self.policy(*batch,
                                    True,
                                    graph_filter=None, filter_key=None,
                                    )
                
                out_policy_li_ = out_policy
                out_value_li_ = out_value
                
                for out_policy, out_value, action, state in zip(out_policy_li_, out_value_li_,old_actions, old_states):
                    action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
                    ind = state.action_space.index(action)
                    dist = Categorical(action_prob)
                    action_logprob = dist.log_prob(torch.tensor(ind).to(device))
                    # dist_entropy = dist.entropy()
                    state_value = out_value

                    action_logprobs_li.append(action_logprob)
                    dist_entropy_li.append(dist.entropy())
                    state_values_li.append(state_value)
            
                logprobs,state_values,dist_entropy = torch.stack(action_logprobs_li), torch.stack(state_values_li), torch.stack(dist_entropy_li)
        
                

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - torch.stack(old_logprobs))
                advantages = torch.stack(advantages)
                # Finding Surrogate Loss   
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss_policy = -torch.min(surr1, surr2).mean()
                rewards = torch.stack(rewards)
                loss_value = 0.5 * self.MseLoss(state_values, rewards)
                loss_dist_entropy = - 0.01 * dist_entropy.mean()
                loss = (loss_policy + loss_value + loss_dist_entropy)/iter_num
                loss.mean().backward()

                loss_policy_li.append(loss_policy)
                loss_value_li.append(loss_value)
                loss_dist_entropy_li.append(loss_dist_entropy)
                
            
            # take gradient step
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            end_time = time.time() 
            if len(loss_policy_li) == 1:
                loss_policy = loss_policy_li[0]
                loss_value = loss_value_li[0]
                loss_dist_entropy = loss_dist_entropy_li[0]
            else:
                loss_policy = torch.stack(loss_policy_li,dim=0).mean()
                loss_value = torch.stack(loss_value_li,dim=0).mean()
                loss_dist_entropy = torch.stack(loss_dist_entropy_li,dim=0).mean()
            execution_time = end_time - start_time  # 计算执行时间
            # print(f"程序执行时间: {execution_time} 秒")
            # loss.mean().item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return loss.mean().item(),loss_policy.item(),loss_value.item(),loss_dist_entropy.item()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        

################ PPO hyperparameters ################


update_timestep = 3 * 0   # update policy every n timesteps
K_epochs = 3        # update policy for K epochs
eps_clip = 0.1             # clip parameter for PPO
gamma = 0.99         # discount factor
batch_size = 800
max_training_episodes = int(1e6)

random_seed = 0         # set random seed if required (0 = no random seed)

saverP = ParameterSaver()
if imitationlearning is True:
    checkpoint_path_para = checkpoint_path
else:
    checkpoint_path_para = None
if lr_decay is True:
    T_max_para = T_max
else:
    T_max_para = None

parameters = {
    'pid': os.getpid(),
    'file_path':os.path.abspath(__file__),
    'learning_rate': lr,
    'checkpoint_path': checkpoint_path_para,
    'T_max' : T_max_para,
    'gamma':gamma,
    'weight_decay':weight_decay

}
saverP.save(parameters,file_name=f'{timestamp}.log')

###################### logging ######################
#### log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + timestamp + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)


#### create new log file for each run 
log_f_name = log_dir + '/PPO_' + timestamp + "_log.txt"

print("current logging run number for " + timestamp)
print("logging at : " + log_f_name)

############# print all hyperparameters #############
log_f = open(log_f_name,"w+")
log_f.write("--------------------------------------------------------------------------------------------\n")


log_f.write("PPO update frequency : " + str(update_timestep) + " timesteps\n") 
log_f.write(f"PPO K epochs : {K_epochs}\n")
log_f.write(f"PPO epsilon clip : {eps_clip}\n")
log_f.write(f"discount factor (gamma) : {gamma}\n", )
log_f.write(f"learning rate : {lr}\n")
log_f.write("--------------------------------------------------------------------------------------------\n")
log_f.write(f"d_enc : {FLAGS.d_enc}\n")
log_f.write(f"device : {FLAGS.device}\n")
if imitationlearning is True:
    log_f.write(f"im_ckpt_path : {checkpoint_path}\n")

log_f.write("--------------------------------------------------------------------------------------------\n")

def env_step(state,action):
    nn_mapping = state.nn_mapping.copy()
    nn_mapping[action[0]] = action[1]
    # state.pruned_space.append((action[0],action[1]))
    state.ori_candidates[:,action[1]] = False
    state.ori_candidates[action[0],:] = False
    new_state = State(state.g1,state.g2,
                    nn_mapping=nn_mapping,
                    g1_reverse=state.g1_reverse,
                    g2_reverse=state.g2_reverse,
                    ori_candidates=state.ori_candidates)
    reward = get_reward(action[0],action[1],state)
    # update_state(new_state,self.threshold)
    if len(nn_mapping) == len(state.g1.nodes):
        return new_state,state,reward,True
    else:
        return new_state,state,reward,False

def init_process(env_input,model, device, shared_array):
    buffer = RolloutBuffer()
    costs = []
    for state_init,ord in env_input:
        stack = [state_init]    
        while stack:
            state = stack.pop()
            with torch.no_grad():
                state.action_space = state.get_action_space(ord)
                pre_processed = _preprocess_NSUBS(state, 0)
                batch, data = create_batch_data([pre_processed])
                batch = batch[:-1]
                model.eval()
                out_policy, out_value, = \
                    model(*batch,
                        True,
                        graph_filter=None, filter_key=None,
                        )
                action_prob = F.softmax(out_policy[0] - out_policy[0].max()) + 1e-10
                m = Categorical(action_prob)
                action_ind = m.sample()
                action = state.action_space[action_ind]

                buffer.gid_pairs.append((state.g1.graph['gid'],state.g2.graph['gid']))
                buffer.mappings.append(state.nn_mapping)
                buffer.actions.append(action)
                buffer.logprobs.append(m.log_prob(action_ind).item())
                buffer.state_values.append(out_value[0].item())

                newstate, state,reward, done = env_step(state,action)
                buffer.rewards.append(reward)
                buffer.is_terminals.append(done)
                stack.append(newstate)
                if done:
                    cost = calculate_cost(newstate.g1,newstate.g2,newstate.nn_mapping)
                    costs.append(cost)
                    break
    average_cost = sum(costs)/len(costs)
    buffer_bytes = pickle.dumps((buffer,average_cost))

    bytes_per_element = sys.getsizeof(shared_array[0])

    # 计算总字节数
    total_bytes = len(shared_array) * bytes_per_element

    assert len(buffer_bytes) <= total_bytes
    np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8)
    np_array[:len(buffer_bytes)] = np.frombuffer(buffer_bytes, dtype=np.uint8)
    # np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8)
    # np_array[:len(data)] = np.frombuffer(data, dtype=np.uint8)
    # queue.put((buffer_bytes,cost))
    # queue.put(action)

def collect_data(ppo_agent,env,update_timestep):
    # mp.set_start_method('spawn') # 或者 'forkserver'
    env_li = []
    shared_memorys= []
    for i in range(3):
        state_init_ord_li = []
        for j in range(int(update_timestep//3)):
            state_init = env.reset()
            ord = env.order
            state_init_ord_li.append((state_init,ord))
        env_li.append(state_init_ord_li)
        shared_memorys.append(multiprocessing.Array('B', 1024 * 1024*80))
    processes = []
    for env_input,shared_memory in zip(env_li,shared_memorys):
        p = mp.Process(target=init_process, args=(env_input,ppo_agent.policy_old, device, shared_memory)) 
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    results = []
    avg_costs = []
    for shared_memory in shared_memorys:
        np_array = np.frombuffer(shared_memory.get_obj(), dtype=np.uint8)
        buffer_bytes = np_array.tobytes()
        buffer,avg_cost = pickle.loads(buffer_bytes)
        avg_costs.append(avg_cost)
        results.append(buffer)
        del shared_memory
    average_cost = sum(avg_costs)/len(avg_costs)
    return results,average_cost

# def collect_data(ppo_agent,env):
#     # mp.set_start_method('spawn') # 或者 'forkserver'
#     env_li = []
#     for i in range(2):
#         state_init = env.reset()
#         ord = env.order
#         env_li.append((state_init,ord))
#     threads = []
#     # queue = mp.Queue()
#     results_queue = queue.Queue()
#     for state_init,ord in env_li:
#         thread = threading.Thread(target=init_process, args=(ord,state_init,ppo_agent.policy_old, device, results_queue)) 
#         thread.start()
#         threads.append(thread)
#     for thread in threads:
#         thread.join()
#     results = []
#     while not results_queue.empty():
#         results.append(results_queue.get())

#     return results






def main():
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 或者处理已经设置的情况
    writer = SummaryWriter(f'plt_RL/{timestamp}')
    env  = environment(dataset)
    ppo_agent = PPO(lr, gamma, K_epochs, eps_clip,batch_size)
   

    for episode in range(max_training_episodes):
        print_open_files_count()
        results,cost = collect_data(ppo_agent,env,update_timestep)
        print(cost)
        buffers = results
        # buffers = [res[0] for res in result]
        # costs = [res[1] for res in result]

        # cost_avg = sum(costs)/len(costs)
        # print(f"cost_avg: {cost_avg}")

        for buffer in buffers:
            buffer.dataset = dataset
            buffer.info2state()
            ppo_agent.buffer += buffer 

        print_open_files_count()
        # if episode % update_timestep == 0:
        # loss,loss_policy,loss_value,loss_dist_entropy = ppo_agent.update()

        # writer.add_scalar('Cost', cost, episode)
        # writer.add_scalar('Loss/total', loss, episode)
        # writer.add_scalar('Loss/policy', loss_policy, episode)
        # writer.add_scalar('Loss/value', loss_value, episode)
        # writer.add_scalar('Loss/dist_entropy', loss_dist_entropy, episode)


            
            
        if lr_decay:
            for _ in range(update_timestep):
                ppo_agent.scheduler.step()

        if episode % 10 == 0:
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
        # writer.close()
    

if __name__ == '__main__':
    main()
    log_f.close()
    



