import torch
import torch.nn as nn
from collections import deque
import random
import numpy as np
import pickle
from collections import namedtuple

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('pov', 'action', 'reward', 'done', 'next'))

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        samples = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in samples]
        
        return batch

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path):
        index, size, transitions \
             = pickle.load(open(path, "rb"))
       
        for i in range(index-1):
            trans=Transition(transitions[i][0],transitions[i][1], transitions[i][2], transitions[i][3], transitions[i+1][0])
            self.add(trans)
