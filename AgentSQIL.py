import torch
from torch.distributions import Categorical
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from Network import NetworkSQIL
import matplotlib.pyplot as plt
import cv2
import Buffer, MinecraftWrapper
import gym
import minerl

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgentSQIL(nn.Module):
    def __init__(self, input_channels, n_actions):
        onlineQNetwork = NetworkSQIL(input_channels, n_actions).to(device)
        targetQNetwork = NetworkSQIL(input_channels, n_actions).to(device)
        targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

        self.optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)
        
        GAMMA = 0.9
        REPLAY_MEMORY = 300000 
        BATCH = 16
        UPDATE_STEPS = 8
        THRESHOLD = 2600 
        expert_memory_replay = Buffer(REPLAY_MEMORY)
        expert_memory_replay.load("Dataset_chopTree.pkl")
    


    online_memory_replay = Buffer(25000)
    
    
    def sqil():
            
        learn_steps = 0
        begin_learn = False
        episode_reward = 0
        env_super = gym.make("MineRLTreechop-v0")
        env = MinecraftWrapper(env_super, action_manager)
        state = env.reset()
        losses=[]
        td_errors=[]
        running_td=0
        for epoch in range(1):
            print("EPOCH",epoch)
            state = env.reset()
            episode_reward = 0
            curr_t=0
            running_loss=0
            for time_steps in range(1500000):
                curr_t+=1
                statet = state.detach().numpy()
                try:action = onlineQNetwork.choose_action(statet)
                except BaseException:print(time_steps,"state",statet)
                next_state, reward, done = env.step(action)
                episode_reward += reward
                online_memory_replay.add(Transition(state, action, 0, done, next_state))#changed so that rew is 0 for sampled experiences
                #state = next_state

                #if online_memory_replay.size() > THRESHOLD:
                if curr_t > THRESHOLD:
                    if begin_learn is False:
                        print('learn begins!')
                        begin_learn = True
                    learn_steps += 1
                    if learn_steps % UPDATE_STEPS == 0:
                        targetQNetwork.load_state_dict(onlineQNetwork.state_dict())

                    online_batch = online_memory_replay.sample(BATCH//2, False)
                    online_batch_state, online_batch_action, online_batch_reward, online_batch_done, online_batch_next_state = zip(*online_batch)
                    
                    online_batch_state = torch.stack((online_batch_state[0], online_batch_state[1], online_batch_state[2], online_batch_state[3], online_batch_state[4], online_batch_state[5], online_batch_state[6], online_batch_state[7]))
                    online_batch_next_state = torch.stack((online_batch_next_state[0], online_batch_next_state[1], online_batch_next_state[2], online_batch_next_state[3], online_batch_next_state[4], online_batch_next_state[5], online_batch_next_state[6], online_batch_next_state[7]))
                    online_batch_action = prepro(online_batch_action, device)
                    online_batch_reward = prepro(online_batch_reward, device)
                    online_batch_done = prepro(online_batch_done, device)

                    expert_batch = expert_memory_replay.sample(BATCH//2, False)
                    expert_batch_state, expert_batch_action, expert_batch_reward, expert_batch_done, expert_batch_next_state = zip(*expert_batch)
                    expert_batch_state = torch.permute(torch.stack((expert_batch_state[0], expert_batch_state[1], expert_batch_state[2], expert_batch_state[3], expert_batch_state[4], expert_batch_state[5], expert_batch_state[6], expert_batch_state[7])), [0,3,2,1])
                    expert_batch_next_state = torch.permute(torch.stack((expert_batch_next_state[0], expert_batch_next_state[1], expert_batch_next_state[2], expert_batch_next_state[3], expert_batch_next_state[4], expert_batch_next_state[5], expert_batch_next_state[6], expert_batch_next_state[7])), [0,3,2,1])
                    expert_batch_action = prepro(expert_batch_action, device)
                    expert_batch_reward = prepro(torch.ones(8), device)#is this right?
                    expert_batch_done = prepro(expert_batch_done, device)
                   
                    temp1 = torch.permute(expert_batch_next_state.type(torch.float32), [0,3,1,2])
                    temp2 = torch.permute(expert_batch_state.type(torch.float32), [0,3,1,2])
                    next_q = targetQNetwork(temp1.to(device))
                    next_v = targetQNetwork.getV(next_q)
                    #fix this
                    y_expert = torch.sum((onlineQNetwork(temp2.to(device)).gather(1, expert_batch_action.long()) - expert_batch_reward + (1 - expert_batch_done) * GAMMA * next_v)**2)/8

                    temp1 = torch.permute(online_batch_next_state.type(torch.float32), [0,3,1,2])
                    temp2 = torch.permute(online_batch_state.type(torch.float32), [0,3,1,2])
                    next_q = targetQNetwork(temp1.to(device))
                    next_v = targetQNetwork.getV(next_q)
                    #and this
                    y_online = torch.sum((onlineQNetwork(temp2.to(device)).gather(1, online_batch_action.long())-  online_batch_reward + (1 - online_batch_done) * GAMMA * next_v)**2)/8
                    td_errors.append(y_online)
                    running_td+=y_online
                    loss=y_expert + 0.001*(y_online)
                    running_loss+=loss.item()
                    if(time_steps%2500==0): 
                        td_errors.append(running_td/2500)
                        losses.append(running_loss/2500)
                        print(time_steps, loss, "running loss:", running_loss/2500,"RUNnning td_error:",running_td/2500)
                        running_loss=0
                        running_td=0

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if done:
                    print(time_steps, "DONE",done)
                    env.reset()
                
                state = next_state

                if time_steps % 25000 == 0:
                    torch.save(onlineQNetwork.state_dict(), str(time_steps)+'_pls_work.para')
                    
                    print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
            plt.plot(td_errors)
            plt.show()
            plt.savefig('td_errors.png')



    def evaluate_agentSQIL(agent, action_manager, steps=7500):
        state=env.reset()
        curr_reward=0
        cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
        replay=[]
        for i in range(steps):
            action=agent.choose_action(state.float(), exploit=True)
            state, reward, done=env.step(action)
            if done: break
            replay.append(state.numpy())
            if(reward>0): 
                print(i,reward)
                curr_reward+=reward
            #if(action!=82):print(action)
        
            cv2.imshow("Input", state.numpy())
            cv2.waitKey(20)
            
        print("OBTAINED ", curr_reward, "IN", i, "STEPS")
