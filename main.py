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
from AgentSQIL import AgentSQIL
from MinecraftWrapper import MinecraftWrapper
import matplotlib.pyplot as plt
import argparse


def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--agent', type=str, default="BC")
    parser.add_argument('--task', type=str, default="chopTree")
    parser.add_argument('--create_db', action=argparse.BooleanOptionalAction)
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction)
    parser.add_argument('--help', action=argparse.BooleanOptionalAction)
    return parser.parse_args(raw_args)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_agent(task, agent_type, dataset,save_dir="C:\\Users\\andre\\Desktop\\MineLearn\\models\\"):
    epochs=10 
    batch_size=32
    learning_rate=0.0000625
    db=Dataset(ActionManager(),capacity=400000, device=device)

    if(agent_type=="SQIL"):
        agent=AgentSQIL(n_actions=112, input_channels=3)
        agent.sqil()
    elif (agent_type=="BC"): 
        if(task=="chopTree"):
            agent=Agent(112, 3, batch_size, learning_rate, device, train=True, vec=False)
            db.load("Dataset_"+task+".pkl")
            agent.behaviour_cloning(db,epochs=epochs,train_split=1.0)
        elif(task=="digStone"):
            agent=Agent(130, 3, batch_size, learning_rate, device, train=True, vec=False)
            db.load("Dataset_"+task+".pkl")
            agent.behaviour_cloning(db,epochs=epochs,train_split=1.0)
        elif(task=="craftWoodenPickaxe"):
            agent=Agent(130, 3, batch_size, learning_rate, device, train=True, vec=True)
            db.load("Dataset_"+task+".pkl")
            agent.behaviour_cloning(db,epochs=epochs,train_split=1.0)
        elif(task=="craftStonePickaxe"):
            agent=Agent(130, 3, batch_size, learning_rate, device, train=True, vec=True)
            db.load("Dataset_"+task+".pkl")
            agent.behaviour_cloning(db,epochs=epochs,train_split=1.0)


    agent.save(save_dir, task+"_model")

    print("finished TRAINING")


def create_db(task):
    if task=='chopTree': action_manager=ActionManager(full=False)
    else: action_manager=ActionManager(full=True)
    db=Dataset(action_manager,capacity=400000, device=device)
    put_data_into_dataset(task=task,dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
    db.save("Dataset_"+task+".pkl")

def evaluate_agents():
    action_manager=ActionManager(device, full=False)
    num_actions = action_manager.num_action_ids_list[0]   
    batch_size=32
    learning_rate=0.0000625
    agent = Agent(num_actions, 3, batch_size, learning_rate, device, train=False)
    
    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="treechop")
    env_super=gym.make("MineRLObtainDiamond-v0")
    env_super.seed(16)
    env=MinecraftWrapper(env_super, action_manager)
    for i in range(20):
        print(i)
        agent.evaluate_agent(env, agent, i)

def main(args):
    if args.help: print("use --agent to specify which type of agente you want to train, possible values: SQIL, BC\n \
        use --task to specify the task, possible values: chopTree, craftWoodenPickaxe, digStone, craftStonePickaxe \n \
        add --create_db if you want to create a new dataset, you must specify the value of --task \n \
        add --train if you want to train an agent, you must specify --agent and --task. To modify training parameters you have to modifiy the values in the script\n\
        add --evaluate to run evaluation on the trained agents. You have to specify --agent")
    if args.evaluate: evaluate_agents()
    if args.create_db: create_db(args.task)
    if args.train: train_agent(args.task, args.agent)


if __name__=='__main__':
    main(get_args())

