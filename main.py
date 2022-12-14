import minerl
from minerl.data import BufferedBatchIter
from dataset import Dataset
from data_manager import ActionManager
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

def train_agent(agent, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", epochs=12, train_split=0.8):
    agent.behaviour_cloning(dataset,epochs=epochs,train_split=train_split)

    agent.save(save_dir, str(epochs))

    print("finished TRAINING")

def evaluate_agent(agent, action_manager, steps):
    print("EVALUATION")
    env_super=gym.make("MineRLObtainDiamond-v0")
    print("ENV CREATED")
    #change the seed to modify the word
    env_super.seed(5)
    env=MinecraftWrapper(env_super, action_manager)
    state=env.reset()
    print(state)
    steps=0
    curr_reward=0
    while(True):
        steps+=1
        action=agent.act(state.to(device))
        #print(action)
        state, reward, done=env.step(action)
        curr_reward+=reward
        if(reward>0): print(reward)
        plt_state=state.to('cpu').numpy()
        cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
        cv2.imshow("Input", plt_state)
        cv2.waitKey(50)
    print("OBTAINED ", curr_reward, "IN", steps, "STEPS")


if __name__=='__main__':
 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_manager=ActionManager(device)
    num_actions = action_manager.num_action_ids_list[0]   
    batch_size=32
    learning_rate=0.0000625
    agent = Agent(num_actions, 3, batch_size, learning_rate, device, train=False)
    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="25")
    evaluate_agent(agent, action_manager, 1000)
    capacity=400001
    db=Dataset(action_manager,capacity=capacity, device=device)
    create_db=True
    if(create_db):
        put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
        db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    
    print("DB LOADED")
    agent = Agent(num_actions, 3, batch_size, learning_rate, device)
    print("READY TO TRAIN")
    train_agent(agent, db,epochs=12,train_split=0.8)

