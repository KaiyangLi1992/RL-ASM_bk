import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.dvn_wrapper import create_dvn
from create_batch import _preprocess_NSUBS,create_batch_data

device = FLAGS.device
dim = FLAGS.dim

def _create_model(d_in_raw):
    model = create_dvn(d_in_raw, FLAGS.d_enc)
    return model.to(FLAGS.device)

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
    def __init__(self,config,checkpoint):
        self.gamma = config.gamma
        self.eps_clip = config.eps_clip
        self.K_epochs = config.K_epochs
        self.lr = config.lr
        self.buffer = RolloutBuffer()
        self.batch_size = config.batch_size

        self.policy = _create_model(dim).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), self.lr,weight_decay=config.weight_decay) 
   
        if config.imitationlearning is True:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

        self.policy_old = _create_model(dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        if config.lr_decay:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.T_max)

    def compute_discounted_rewards(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return rewards
    
    def select_action(self,state,env):
            with torch.no_grad():
                state.action_space = state.get_action_space(env.order)
                pre_processed = _preprocess_NSUBS(state,0)
                batch, data = create_batch_data([pre_processed],1,None,False)
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
    
    def compute_advantages(self, rewards, old_state_values):
        advantages = rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        return advantages

    def preprocess_states(self, old_states):
        pre_processed_li = [_preprocess_NSUBS(s, 0) for s in old_states]
        return pre_processed_li
    
    def calculate_loss(self,batch,iter_num):
        action_logprobs_li = []
        dist_entropy_li= []
        state_values_li = []

        old_datas = batch[-5:]
        batch = batch[:-6]

        old_logprobs,advantages,rewards,old_states,old_actions= old_datas

        ######### train or eval 
        self.policy.train()
        out_policys, out_values = \
                self.policy(*batch,
                            True,
                            graph_filter=None, filter_key=None,
                            )
        for out_policy, out_value, action, state in zip(out_policys, out_values,old_actions, old_states):
            action_prob = F.softmax(out_policy - out_policy.max()) + 1e-10
            ind = state.action_space.index(action)
            dist = Categorical(action_prob)
            action_logprob = dist.log_prob(torch.tensor(ind).to(device))
            state_value = out_value

            action_logprobs_li.append(action_logprob)
            dist_entropy_li.append(dist.entropy())
            state_values_li.append(state_value)
            
        logprobs,state_values,dist_entropys = torch.stack(action_logprobs_li), torch.stack(state_values_li), torch.stack(dist_entropy_li)
        state_values = torch.squeeze(state_values)
                
        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = torch.exp(logprobs - torch.stack(old_logprobs))
        advantages = torch.stack(advantages) 
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        loss_policy = -torch.min(surr1, surr2).mean()/iter_num


        rewards = torch.stack(rewards)
        loss_value = 0.5 * nn.MSELoss()(state_values, rewards)/iter_num
        loss_dist_entropy = - 0.01 * dist_entropys.mean()/iter_num
        loss = loss_policy + loss_value + loss_dist_entropy
        loss.backward()

        return loss,loss_policy,loss_value,loss_dist_entropy

    
        


      
    def update(self):
        rewards = self.compute_discounted_rewards()
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)


        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = self.compute_advantages(rewards, old_state_values)
        pre_processed_li = self.preprocess_states(old_states)

        
        old_data = (old_logprobs,advantages,rewards,old_states,old_actions)
        batch, dataloader = create_batch_data(pre_processed_li,self.batch_size,old_data,include_label=True)
        iter_num = int(len(old_states)/self.batch_size)+1
    


        for _ in range(self.K_epochs):
            loss_li,loss_policy_li,loss_value_li,loss_dist_entropy_li = [],[],[],[] 
            # logprobs, state_values, dist_entropy = self.policy_evaluate(old_states, old_actions,self.buffer.is_terminals)
            for step, batch in enumerate(dataloader):
                loss,loss_policy,loss_value,loss_dist_entropy = self.calculate_loss(batch,iter_num)
                loss_li.append(loss)
                loss_policy_li.append(loss_policy)
                loss_value_li.append(loss_value)
                loss_dist_entropy_li.append(loss_dist_entropy)
                
            
            # take gradient step
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            if len(loss_policy_li) == 1:
                loss = loss_li[0]
                loss_policy = loss_policy_li[0]
                loss_value = loss_value_li[0]
                loss_dist_entropy = loss_dist_entropy_li[0]
            else:
                loss = torch.stack(loss_li,dim=0).sum()
                loss_policy = torch.stack(loss_policy_li,dim=0).sum()
                loss_value = torch.stack(loss_value_li,dim=0).sum()
                loss_dist_entropy = torch.stack(loss_dist_entropy_li,dim=0).sum()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return loss.mean().item(),loss_policy.item(),loss_value.item(),loss_dist_entropy.item()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        