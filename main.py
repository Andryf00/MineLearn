import minerl
from minerl.data import BufferedBatchIter
from dataset import Dataset
from data_manager import ActionManager
from load_dataset import put_data_into_dataset
import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    action_manager=ActionManager(torch.device('cpu'))
    db=Dataset(action_manager,capacity=5000)
    if(False):
        put_data_into_dataset(dataset=db, action_manager=action_manager, minecraft_human_data_dir="C:\\Users\\andre\\Desktop\\MineLearn\\")
        db.save("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\50k_db.pkl")
    db.load("C:\\Users\\andre\\Desktop\\MineLearn\\Dataset\\50k_db.pkl")
    print("DB LOADED")
    #print(db.transitions[0])
    print(db.transitions[0][1],db.transitions[0][2])
    plt.imshow(db.transitions[0][0].permute(1,2,0).numpy())
    #plt.block()
