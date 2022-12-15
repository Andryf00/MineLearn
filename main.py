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
from time import sleep

def train_agent(agent, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", epochs=10, train_split=0.8):
    agent.behaviour_cloning(dataset,epochs=epochs,train_split=train_split)

    agent.save(save_dir, str(epochs))

    print("finished TRAINING")


"""agent = Agent(130, 3, 32, 0.1, torch.device('cuda'),vec=True)
agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="crafting")

action=agent.act(torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).to(torch.device('cuda')), True)            
"""
def evaluate_agent(env, agent, action_manager, steps):
    import torch
    print("EVALUATION")
    env=env
    state=env.reset()
    steps=0
    curr_reward=0
    crafting=False
    while(curr_reward<16 and steps<10000):
        steps+=1
        action=agent.act(state.to(device), crafting)
        #print(action)
        
        state, reward, done=env.step(action,crafting)
        curr_reward+=reward
        plt_state=state['pov']
        #print("logs:",state['inventory']['log'])
        #if(steps>50):
            #crafting=True
        if(state['inventory']['log']>16 and not crafting):
            print("CRAFTING BOYZ")
            agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
            agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="250")
            crafting=True
        if not crafting:
            state=torch.from_numpy(state['pov'].copy())
        else: 
            list_inv=[x.item() for x in list(state['inventory'].values())]
            #print(list_inv)
            state=torch.Tensor(list_inv)
        if(reward>0): print(reward)
        """
            cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("Input", plt_state)
        cv2.waitKey(50)"""
    print("OBTAINED ", curr_reward, "IN", steps, "STEPS")


if __name__=='__main__':
 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_manager=ActionManager(device)
    num_actions = action_manager.num_action_ids_list[0]   
    batch_size=32
    learning_rate=0.0000625
    
    agent = Agent(num_actions, 3, batch_size, learning_rate, device, train=False)
    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="25")
    env_super=gym.make("MineRLObtainDiamond-v0")
    env=MinecraftWrapper(env_super, action_manager)
    print("ENV CREATED")
    #change the seed to modify the word
    env_super.seed(22)
    for i in range(10):
        print(i)
        evaluate_agent(env, agent, action_manager, 1000)
    capacity=1819
    db=Dataset(action_manager,capacity=capacity, device=device)
    create_db=True
    if(False):
        put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
        db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    #for t in db.transitions:
    #    print(t[0].size())
    print("DB LOADED")
    agent = Agent(num_actions, 3, batch_size, 0.0001, device,vec=True)
    print("READY TO TRAIN")
    #train_agent(agent, db,epochs=250,train_split=1.0)

