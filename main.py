import minerl
from minerl.data import BufferedBatchIter
from dataset import Dataset
from data_manager import ActionManager, ActionManager2
from load_dataset import put_data_into_dataset
from Agent import Agent
import torch
import matplotlib.pyplot as plt
import gym
import time
from MinecraftWrapper import MinecraftWrapper
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from time import sleep

def train_agent(agent, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", epochs=10, train_split=0.8):
    agent.behaviour_cloning(dataset,epochs=epochs,train_split=train_split)

    agent.save(save_dir, "craftingstone2layers_2")

    print("finished TRAINING")

def evaluate_agent_SQIL(env, agent, action_manager, iter):
    import torch
    env_super=gym.make("MineRLObtainDiamond-v0")
    env=MinecraftWrapper(env_super, action_manager)
    print("ENV CREATED")
    #change the seed to modify the word
    env_super.seed(22)
    print("EVALUATION")
    env=env
    state=env.reset()
    done=False
    steps=0
    curr_reward=0
    crafting=False
    replay=[]
    while(curr_reward<16 and steps<10000 and not done):
        #cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
        steps+=1
        action=agent.act(state.to(device), crafting)
        #print(action)
        
        state, reward, done=env.step(action,crafting)
        curr_reward+=reward
        plt_state=state['pov']
        """
        if(iter==0):
            #cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("Input", plt_state)
            cv2.waitKey(50)"""
        replay.append(plt_state)
        if(state['inventory']['log']>16 and not crafting):
            print("CRAFTING BOYZ")
            agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
            agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="crafting")
            crafting=True
        if not crafting:
            state=torch.from_numpy(state['pov'].copy())
        else: 
            list_inv=[x.item() for x in list(state['inventory'].values())]
            #print(list_inv)
            state=torch.Tensor(list_inv)
        if(reward>0): print(reward)
    
    crafting=False


def evaluate_agent(env, agent, action_manager, iter):
    import torch
    env=env
    state=env.reset()
    done=False
    steps=0
    curr_reward=0
    crafting=False
    replay=[]
    while(curr_reward<16 and steps<10000 and not done):
        cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
        steps+=1
        action=agent.act(state.to(device), crafting)
        #print(action)
        
        state, reward, done=env.step(action,crafting)
        curr_reward+=reward
        plt_state=state['pov']
        """
        if(iter==0):
            #cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("Input", plt_state)
            cv2.waitKey(50)"""
        replay.append(plt_state)
        if(state['inventory']['log']>16 and not crafting):
            print("CRAFTING BOYZ")
            agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
            agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="craftingwood2layers")
            crafting=True
        if not crafting:
            state=torch.from_numpy(state['pov'].copy())
        else: 
            list_inv=[x.item() for x in list(state['inventory'].values())]
            #print(list_inv)
            state=torch.Tensor(list_inv)
        if(reward>0): print(steps, reward)
    
    crafting=False       
    if not done:
        state, reward, done=env.step(0, True)
        #print(state['equipped_items'])
        state=torch.from_numpy(state['pov'].copy())
        
        agent = Agent(130, 3, batch_size, learning_rate, device,vec=False)
        agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="11")
        while(steps<10000 and not done):
            steps+=1
            action=agent.act(state.to(device), crafting)
            state, reward, done=env.step(action,True)
            last_state=state
            plt_state=state['pov']
            replay.append(plt_state)
            if(state['inventory']['cobblestone']>10 and not crafting):
                agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
                agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="craftingstone2layers_2")
                crafting=True
            if not crafting:
                state=torch.from_numpy(state['pov'].copy())
            else: 
                list_inv=[x.item() for x in list(state['inventory'].values())]
                #print(list_inv)
                state=torch.Tensor(list_inv)
            curr_reward+=reward
            if(reward>0): print(steps, reward)
            """if(iter==0):
                #cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
                cv2.imshow("Input", plt_state)
                cv2.waitKey(50)"""
            if(curr_reward>95):
                break
        if not done and curr_reward>95:
            state=torch.from_numpy(last_state['pov'].copy())
            crafting=False
            agent = Agent(130, 3, batch_size, learning_rate, device,vec=False)
            agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="11")
            while(steps<15000 and not done ):
                steps+=1
                action=agent.act(state.to(device), crafting)
                state, reward, done=env.step(action,True)
                last_state=state
                plt_state=state['pov']
                replay.append(plt_state)
                curr_reward+=reward
                if(reward>0): print(steps, reward)
                state=torch.from_numpy(state['pov'].copy())
                
    print("OBTAINED ", curr_reward, "IN", steps, "STEPS")
    #if(curr_reward>90):
        #print("Final inv:", last_state['inventory'])    
        #print("saving replay")
        #pickle.dump(replay, open('C:\\Users\\andre\\Desktop\\MineLearn\\replay\\'+str(steps)+'_'+str(iter)+'_'+str(curr_reward)+'.pkl', 'wb'))


def create_db(capacity,action_manager, database):
    db=Dataset(action_manager,capacity=capacity, device=device)
    put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
    db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\crafting_stone_db.pkl")


if __name__=='__main__':
 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_manager=ActionManager2(device)
    num_actions = action_manager.num_action_ids_list[0]   
    batch_size=32
    learning_rate=0.0000625
    agent = Agent(112, 3, batch_size, learning_rate, device, train=False)
    #agent = Agent(130, 3, batch_size, learning_rate, device, train=True, vec=True)
    
    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="treechop")
    env_super=gym.make("MineRLObtainDiamond-v0")
    env=MinecraftWrapper(env_super, action_manager)
    print("ENV CREATED")
    #change the seed to modify the word
    env_super.seed(22)
    env_super=gym.make("MineRLObtainDiamond-v0")
    env=MinecraftWrapper(env_super, action_manager)
    
    print("ENV CREATED")
    #change the seed to modify the word
    env_super.seed(22)
    
    for i in range(20):
        print(i)
        evaluate_agent(env, agent, action_manager, i)
    capacity=400000
    db=Dataset(action_manager,capacity=capacity, device=device)
    if(False):
        create_db(capacity=1000,action_manager=action_manager,database=db)
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\crafting_stone_db.pkl")
    print("DB LOADED")
    print("READY TO TRAIN")
    #train_agent(agent, db,epochs=100,train_split=1.0)

