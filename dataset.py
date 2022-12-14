import gym
import minerl
from minerl.data import BufferedBatchIter
import random
from collections import namedtuple
import numpy as np
import torch
import pickle
from operator import itemgetter 


Transition = namedtuple('Transition', ('pov', 'action', 'reward', 'done'))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset():
    def __init__(self,action_manager, device,capacity=1000):
      self.capacity=capacity
      self.size=0
      self.device=device
      self.transitions=np.empty(capacity,dtype=Transition)
        
      self.index=0
      self.action_manager=action_manager

    def append_sample(self, sample):
      state, action, reward, done = sample[0], sample[1], sample[2], sample[4]
      img= state['pov']
      action_id = self.action_manager.get_id(action)
      torch_img = torch.from_numpy(img).permute(2, 0, 1)
      self.transitions[self.index]=Transition(torch_img, action_id, reward, not done)
      self.index+=1

    def append_sample_inv(self, sample):
      state, action, reward, done = sample[0], sample[1], sample[2], sample[4]
      
      inv=list(state['inventory'].values())
      print(inv)
      action_id = self.action_manager.get_id(action)
      torch_vec = torch.Tensor(inv)
      self.transitions[self.index]=Transition(torch_vec, action_id, reward, not done)
      self.index+=1

    def save(self, path):
      print(self.index)
      pickle.dump([self.index, self.size, self.transitions], open(path, 'wb'))

    def load(self, path):
        self.index, self.size, self.transitions \
             = pickle.load(open(path, "rb"))

