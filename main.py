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

def train_agent(agent, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", trainsteps=100):
    for i in range(trainsteps):
        print("training iteration",i)

        agent.learn(i, dataset, write=(i % 1000 == 0))

    agent.save(save_dir, str(trainsteps))

    print("finished TRAINING")

def evaluate_agent(agent, action_manager, steps):
    print("EVALUATION")
    env_super=gym.make("MineRLTreechop-v0")
    print("ENV CREATED")
    env_super.seed(0)
    env=MinecraftWrapper(env_super, action_manager)
    state=env.reset()
    print(state)
    #state=state['pov']
    for i in range(steps):
        action=agent.act(state.to(device))
        print(action)
        state, reward, done=env.step(action)
        plt.imshow(state)
        plt.show()


if __name__=='__main__':
 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_manager=ActionManager(device)
    num_actions = action_manager.num_action_ids_list[0]   
    batch_size=32
    learning_rate=0.0000625
    agent = Agent(num_actions, 3, batch_size, learning_rate, device, train=False)
    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="2000")
    evaluate_agent(agent, action_manager, 1000)
    capacity=420000
    db=Dataset(action_manager,capacity=capacity, device=device)
    create_db=True
    if(create_db):
        put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
        db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\"+str(capacity)+"_db.pkl")
    
    print("DB LOADED")
    agent = Agent(num_actions, 3, batch_size, learning_rate, device)
    print("READY TO TRAIN")
    train_agent(agent, db,trainsteps=2000)

