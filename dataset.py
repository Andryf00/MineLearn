import gym
import minerl
from minerl.data import BufferedBatchIter

from collections import namedtuple
import numpy as np
import torch
import pickle


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))



class Dataset():
    def __init__(self,action_manager, capacity=1000, device=torch.device('cpu')):
      self.capacity=capacity
      self.size=0
      self.device=device
      self.transitions=np.empty(capacity, dtype=[("sample", Transition)])
        
      self.index=0
      self.action_manager=action_manager

    def append_sample(self, sample):
      state, action, reward, done = sample[0], sample[1], sample[2], sample[4]
      img= state['pov']
      action_id = self.action_manager.get_id(action)
      torch_img = torch.from_numpy(img).permute(2, 0, 1)
      self.transitions.append(Transition(torch_img, action_id, reward, not done))
      self.index+=1

    #def get():
    #def sample():
           

    def save(self, path):
        pickle.dump([self.index, self.size, self.transitions], open(path, 'wb'))

    def load(self, path):
        self.index, self.size, self.transitions \
             = pickle.load(open(path, "rb"))

