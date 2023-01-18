import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from Network import Network, vec_Network
import matplotlib.pyplot as plt
from os.path import join as p_join
from torch.utils.data import DataLoader


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

    def act(self, img, vec=False):
        with torch.no_grad():
            if(not vec):
                img=torch.permute(torch.unsqueeze(img,0), [0,3,1,2])
                dim=1
            else: dim=-1
            logits = self.net(img.type(torch.FloatTensor).to(self.device))#, vec)
            #if vec:print("OUTPUT",logits)
            probs = F.softmax(logits,dim).detach().cpu().numpy()
            if(not vec):actions = [np.random.choice(len(p), p=p) for p in probs]
           
            else:
                craft_actions=[x for x in range(112, 130)] 
                actions=[np.random.choice(craft_actions, p=probs[craft_actions]/np.sum(probs[craft_actions]))]
            assert len(actions) == 1

            return actions[0]

    def train_one_epoch(self, epoch_index ,train_size):
        running_loss = 0.
        last_loss = 0.
        for i in range(0,train_size-32,32):
            if self.vec:
                inputs=torch.Tensor(size=[32,18])
                targets=torch.Tensor(size=[32])
            else:
                inputs=torch.Tensor(size=[32,3,64,64])
                targets=torch.Tensor(size=[32])
            for j in range(32):
                try:
                    #print(self.train_dataset[i+j][0])
                    inputs[j]=self.train_dataset[i+j][0]
                    targets[j]=self.train_dataset[i+j][1]  
                except Exception:
                    #print("err")
                    #print(self.train_dataset[i+j])
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
            #if i%(992*1)==0:print("step nr.",i,"curr_loss", running_loss/(i+1))
            if i % 32 == 0:
                last_loss = running_loss / 32 # loss per batch
                if False:print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0
        return last_loss

    def behaviour_cloning(self,dataset, train_split=0.8, epochs=10):
        np.random.shuffle(dataset.transitions)
        train_size=int(dataset.index*train_split)
        print(dataset.index,"*",train_split,"=",train_size)
        self.train_dataset=dataset.transitions[:train_size]
        val_dataset=dataset.transitions[train_size:]
        best_vloss = 1_000_000.
        train_loss=[]
        
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch+1))

            self.train()
            avg_loss=0
            avg_loss = self.train_one_epoch(epoch, train_size)
            print(epoch, avg_loss)
            train_loss.append(avg_loss)
            self.train(False)

            running_vloss = 0.0
            target_tens=torch.LongTensor(1)
            i=0
            #wait why loop? Cant I do batch over the whole dataset?
            for i in range(0):#,dataset.index-train_size): #this causes the gpu to run out of memory but why??
                input, target, r, d=val_dataset[i]
                input=torch.unsqueeze(input,0).to(torch.device('cpu'))
                target_tens[0]=target
                output = self.net(input.type(torch.FloatTensor))
                vloss = F.cross_entropy(output, target_tens)
                running_vloss += vloss

            avg_vloss = running_vloss / (i+1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            print('Training vs. Validation Loss', 'Training', avg_loss, 'Validation' , avg_vloss , epoch + 1)

            # Track best performance, and save the model's state
            if epoch%6==0:
                best_loss = avg_loss
                model_path = 'model_{}_{}'.format(best_loss, epoch)
                torch.save(self.net.state_dict(), model_path)
        print(train_loss)
        plt.plot(train_loss)
        plt.show()
        plt.savefig('treechop2.png')


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