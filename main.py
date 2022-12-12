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


def train_agent(agent, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", trainsteps=100):
    for i in range(trainsteps):
        print("training iteration",i)

        agent.learn(i, dataset, write=(i % 1000 == 0))

    agent.save(save_dir, 'last')

    print("finished TRAINING")

if __name__=='__main__':
    
    action_manager=ActionManager(torch.device('cpu'))
    capacity=200000
    db=Dataset(action_manager,capacity=capacity)
    create_db=False
    if(create_db):
        put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
        db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+capacity+"_db.pkl")
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+capacity+"_db.pkl")
    print("DB LOADED")
    num_actions = action_manager.num_action_ids_list[0]
    batch_size=32
    learning_rate=0.0000625
    device=torch.device('cpu')
    agent = Agent(num_actions, 3, batch_size, learning_rate, device)
    print("READY TO TRAIN")
    train_agent(agent, db)

