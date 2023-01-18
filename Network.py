import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


import math

class vec_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_vec = nn.Linear(in_features= 18,out_features= 256)
        self.fc_vec2 = nn.Linear(in_features= 256,out_features= 130)
    def forward(self, x):
        x=self.fc_vec(x)
        x=F.relu(x)
        x=self.fc_vec2(x)
        x=F.relu(x)
        return x
"""
class vec_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features= 18,out_features= 64)
        self.fc2 =  nn.Linear(in_features= 64,out_features= 130)
    def forward(self, x):
        #print("X",x)
        x=self.fc1(x)
        x=F.relu(x)
        x=self.fc2(x)
        x=F.relu(x)
        return x"""

class Network(nn.Module):
    class _ImpalaResidual(nn.Module):

        def __init__(self, depth):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

        def forward(self, x):
            out = F.relu(x)
            out = self.conv1(out)
            out = F.relu(out)
            out = self.conv2(out)
            return out + x

    def __init__(self, input_channels, n_actions):
        super().__init__()
        depth_in = input_channels
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._ImpalaResidual(depth_out),
                self._ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers, nn.ReLU(),nn.Flatten(),nn.Linear(in_features=4096, out_features=n_actions))
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        
        return self.conv_layers(x)
    
