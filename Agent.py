import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from Network import Network

from os.path import join as p_join
from torch.utils.data import DataLoader


class Agent:
    def __init__(self, num_actions, image_channels, batch_size, learning_rate, device, train=True):
        self.num_actions = num_actions
        #self.writer = writer
        self.device=device
        self.batch_size = batch_size
        """
        self.augment_flip = augment_flip

        self.rev_action_map = None

        if self.augment_flip:
            # flipping the actions horizontally, for the horizontal image flip augmentation:
            self.rev_action_map = [0, 2, 1, 3, 4, 10, 12, 11, 13, 14, 5, 7, 6, 8, 9, 15, 17, 16, 18, 19,
                                   25, 27, 26, 28, 29, 20, 22, 21, 23, 24, 30, 32, 31, 33, 34, 35, 37,
                                   36, 38, 39, 46, 48, 47, 49, 50, 51, 40, 42, 41, 43, 44, 45, 52, 54,
                                   53, 55, 56, 57, 59, 58, 60, 62, 61, 63, 64, 70, 72, 71, 73, 74, 65,
                                   67, 66, 68, 69, 75, 77, 76, 78, 79, 81, 80, 82, 84, 83, 85, 86, 92,
                                   94, 93, 95, 96, 87, 89, 88, 90, 91, 97, 99, 98, 100, 101, 102, 104,
                                   103, 105, 107, 106, 108, 109, 111, 110, 112, 113, 114, 115, 116, 117,
                                   118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
            # this id list can be received by running get_left_right_reversed_mapping() from the ActionManager
        """

        self.net = Network(image_channels, num_actions).to(device=device)
        if(train):
            self.net.train()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)

    def act(self, img):
        with torch.no_grad():
            #print("ACT IMG",img.size())
            img=torch.permute(torch.unsqueeze(img,0), [0,3,1,2])
            #print(img.size())
            logits = self.net(img.type(torch.FloatTensor).to(self.device))#, vec)
            probs = F.softmax(logits, 1).detach().cpu().numpy()

            actions = [np.random.choice(len(p), p=p) for p in probs]#?? sample from prob distribution

            assert len(actions) == 1  # only used with batchsize 1

            return actions[0]

    def train_one_epoch(self, epoch_index ,train_size):
        running_loss = 0.
        last_loss = 0.
        for i in range(0,train_size-32,32):
            inputs=torch.Tensor(size=[32,3,64,64])
            targets=torch.Tensor(size=[32])
            #there must be a better way to do this?
            roba=self.train_dataset[i:i+32]
            print(roba)
            for j in range(32):
                try:
                    inputs[j]=self.train_dataset[i+j][0]
                    targets[j]=self.train_dataset[i+j][1]  
                except Exception:
                    print("err")
                    print(self.train_dataset[i+j])
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
            if i%992*50==0:print("step nr.",i,"curr_loss", running_loss/(i+1))
            """if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0."""
        last_loss=running_loss/i+1
        return last_loss

    def behaviour_cloning(self,dataset, train_split=0.8, epochs=10):
        np.random.shuffle(dataset.transitions)
        train_size=int(dataset.index*train_split)
        print(dataset.index,"*",train_split,"=",train_size)
        self.train_dataset=dataset.transitions[:train_size]
        val_dataset=dataset.transitions[train_size:]
        best_vloss = 1_000_000.
        
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch+1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.train()
            avg_loss=0
            #avg_loss = self.train_one_epoch(epoch, train_size)

            # We don't need gradients on to do reporting
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
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(best_vloss, epoch)
                torch.save(self.net.state_dict(), model_path)



    def learn(self, time_, dataset, write=False):

        states, actions = dataset.sample_state_act(self.batch_size)
        #uncomment if we want data augmentation (flip every image)
        """
        if self.augment_flip:
            if np.random.binomial(n=1, p=0.5):
                states = torch.flip(states, (3,))
                for i in range(actions.shape[0]):
                    actions[i] = self.rev_action_map[actions[i]]
"""      
        print("Learning")
        logits = self.net(states)
        loss = F.cross_entropy(logits, actions)
        print(loss)

        """        if write:
            if self.writer is not None:
                self.writer.add_scalar('loss/cross_entropy', loss.detach().cpu().numpy(), time_)
"""
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