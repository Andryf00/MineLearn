import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from Network import Network, vec_Network
import matplotlib.pyplot as plt
from os.path import join as p_join
from torch.utils.data import DataLoader
import cv2

class Agent:
    def __init__(self, num_actions, image_channels, batch_size, learning_rate, device, train=True, vec=False):
        self.num_actions = num_actions
        #self.writer = writer
        self.device=device
        self.batch_size = batch_size
        self.vec=vec
        if vec:
            self.net=vec_Network().to(device)
        else:
            self.net = Network(image_channels, num_actions).to(device=device)
        if(train):
            self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)

    def act(self, img):
        with torch.no_grad():
            if(not self.vec):
                img=torch.permute(torch.unsqueeze(img,0), [0,3,1,2])
                dim=1
            else: dim=-1
            logits = self.net(img.type(torch.FloatTensor).to(self.device))#, vec)
            probs = F.softmax(logits,dim).detach().cpu().numpy()
            if(not self.vec):actions = [np.random.choice(len(p), p=p) for p in probs]
           
            else:
                craft_actions=[x for x in range(112, 130)] 
                actions=[np.random.choice(craft_actions, p=probs[craft_actions]/np.sum(probs[craft_actions]))]
            assert len(actions) == 1

            return actions[0]

    def train_one_epoch(self, epoch_index ,train_size):
        running_loss = 0.
        last_loss = 0.
        for i in range(0,train_size-32,32):
            #Tensors have different size, depending on wheter we're dealing with RGB images or inventory represented by array
            if self.vec:#inventory
                inputs=torch.Tensor(size=[32,18])
                targets=torch.Tensor(size=[32])
            else:#rgb img
                inputs=torch.Tensor(size=[32,3,64,64])
                targets=torch.Tensor(size=[32])
            for j in range(32):
                try:
                    inputs[j]=self.train_dataset[i+j][0]
                    targets[j]=self.train_dataset[i+j][1]  
                except Exception:#to deal with errors in the dataset
                    inputs[j]=self.train_dataset[i+j-1][0]
                    targets[j]=self.train_dataset[i+j-1][1]                      
                    j-=1
                    i+=1
            self.optimizer.zero_grad()
            outputs = self.net(inputs.to(device=self.device))
            loss = F.cross_entropy(outputs, targets.type(torch.LongTensor).to(self.device))
            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()
            if i%(992*1)==0:print("step nr.",i,"curr_loss", running_loss/(i+1))
            if i % 32 == 0:
                last_loss = running_loss / 32 # loss per batch
                if False:print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        return last_loss

    def behaviour_cloning(self,dataset, train_split=1.0, epochs=10):
        np.random.shuffle(dataset.transitions)
        train_size=int(dataset.index*train_split)
        print(dataset.index,"*",train_split,"=",train_size)
        self.train_dataset=dataset.transitions[:train_size]
        train_loss=[]
        
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch+1))

            self.train()
            avg_loss=0
            avg_loss = self.train_one_epoch(epoch, train_size)
            print(epoch, avg_loss)
            train_loss.append(avg_loss)
            self.train(False)

            print('LOSS train {}'.format(avg_loss))

            # Log the running loss averaged per batch
            # for both training and validation
            print('Training vs. Validation Loss', 'Training', avg_loss, epoch + 1)

            # Track best performance, and save the model's state
            if epoch%6==0:
                best_loss = avg_loss
                model_path = 'model_{}_{}'.format(best_loss, epoch)
                torch.save(self.net.state_dict(), model_path)
        plt.plot(train_loss)
        plt.show()
        plt.savefig('treechop3.png')


    def learn(self, time_, dataset, write=False):

        states, actions = dataset.sample_state_act(self.batch_size)
        print("Learning")
        logits = self.net(states)
        loss = F.cross_entropy(logits, actions)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save(self, path, id_=None):
        
        torch.save(self.net.state_dict(), p_join(path, f'model_{id_}.pth'))
        state = {'optimizer': self.optimizer.state_dict()}
        torch.save(state, p_join(path, f'state_{id_}.pth'))

    def load(self, path, id_=None):
        print("loading")
        if id_ is None:
            self.net.load_state_dict(torch.load(p_join(path, 'model.pth')))
            state = torch.load(p_join(path, 'state.pth'))
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            self.net.load_state_dict(torch.load(p_join(path, f'model_{id_}.pth')))
            state = torch.load(p_join(path, f'state_{id_}.pth'))
            self.optimizer.load_state_dict(state['optimizer'])
        self.net.to(self.device)

    def train(self, bool=True):
        self.net.train(bool)

    def eval(self):
        self.net.eval()

    def evaluate_agent(env, agent, iter):
        import torch
        env=env
        state=env.reset()
        done=False
        steps=0
        curr_reward=0
        crafting=False
        replay=[]
        cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
        while(curr_reward<16 and steps<10000 and not done):
            steps+=1
            action=agent.act(state.to(device), crafting)
            
            state, reward, done=env.step(action,crafting)
            if(reward>0): print(steps, reward)
        
            curr_reward+=reward
            plt_state=state['pov']
            
            #cv2.namedWindow("Input", flags=cv2.WINDOW_NORMAL)
            
            cv2.imshow("Input", plt_state)
            cv2.waitKey(10)
            replay.append(plt_state)
            if(state['inventory']['log']>16 and not crafting):
                print("Loading craftWoodenPickaxe Agent")
                agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
                agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="craftingwood2layers_2")
                crafting=True
            if not crafting:
                
                state=torch.from_numpy(state['pov'].copy())
            else: 
                list_inv=[x.item() for x in list(state['inventory'].values())]
                state=torch.Tensor(list_inv)
            
            
        crafting=False       
        if not done:
            state, reward, done=env.step(0, True) #equipping the wooden pickaxe
            state=torch.from_numpy(state['pov'].copy())
            #loading digStoneAgent
            agent = Agent(130, 3, batch_size, learning_rate, device,vec=False)
            agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="11")
            while(steps<10000 and not done):
                steps+=1
                action=agent.act(state.to(device), crafting)
                state, reward, done=env.step(action,True)
                curr_reward+=reward
                if(reward>0): print(steps, reward)
                plt_state=state['pov']
                replay.append(plt_state)
                if(state['inventory']['cobblestone']>10 and not crafting):
                    #load craftStonePickaxeAgent
                    agent = Agent(num_actions, 3, batch_size, learning_rate, device,vec=True)
                    agent.load(path="C:\\Users\\andre\\Desktop\\MineLearn\\models\\", id_="craftingstone2layers_2")
                    crafting=True
                if not crafting:
                    state=torch.from_numpy(state['pov'].copy())
                else: 
                    list_inv=[x.item() for x in list(state['inventory'].values())]
                    state=torch.Tensor(list_inv)
                cv2.imshow("Input", plt_state)
                cv2.waitKey(50)
                if(curr_reward>95):
                    #stone pickaxe obtained, nothing more to do here
                    break
        print("OBTAINED ", curr_reward, "IN", steps, "STEPS")

