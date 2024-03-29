import os
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle
import sys
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/uclasm/") 
sys.path.append("/home/kli16/esm_NSUBS_RWSE_LapPE/esm/GraphGPS/") 
# Custom module imports
from NSUBS.model.OurSGM.config import FLAGS
from NSUBS.model.OurSGM.saver import ParameterSaver
from environment import environment,calculate_cost
import torch_geometric
from yacs.config import CfgNode as CN
from ppo_batch_update import PPO
torch_geometric.seed_everything(1)







def main():
    with open(f"{FLAGS.dataset}_PPO_config.yaml", 'r') as f:
        yaml_content = f.read()
        cfg = CN.load_cfg(yaml_content)
    with open(cfg.paths.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    ckpt = torch.load(cfg.paths.checkpoint_path,map_location=torch.device(FLAGS.device))

    max_training_episodes = cfg.hyperparameters.max_training_episodes
    update_timestep = cfg.hyperparameters.update_timestep
    lr_decay = cfg.hyperparameters.lr_decay

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    writer = SummaryWriter(f'plt_RL/{timestamp}')
    saverP = ParameterSaver()
    parameters = {
        'pid': os.getpid(),
        'file_path':os.path.abspath(__file__),
        'cfg':yaml_content

    }
    saverP.save(parameters,file_name=f'{timestamp}.log')


    env  = environment(dataset)
    ppo_agent = PPO(cfg.hyperparameters,ckpt)
   
    for episode in range(max_training_episodes):

        state_init = env.reset()
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
    



