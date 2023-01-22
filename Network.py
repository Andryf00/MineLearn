import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


import math


class NetworkSQIL(nn.Module):
    def __init__(self, input_channels, n_actions):
        super().__init__()
        self.alpha = 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.max_pool=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.fc1=nn.Linear(in_features=8192, out_features=1024)
        self.fc2=nn.Linear(in_features=1024, out_features=n_actions)
    
    def forward(self, x):
        x=self.max_pool(F.relu(self.conv1(x)))
        x=self.max_pool(F.relu(self.conv2(x)))
        x=torch.flatten(F.relu(self.conv3(x)),1)
        return F.relu(self.fc2(F.relu(self.fc1(x))))

    def getV(self, q_value):
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value/self.alpha), dim=1, keepdim=True))
        return v
        
    def choose_action(self, state, exploit=False):
        state = torch.permute(torch.unsqueeze(torch.FloatTensor(state),0), [0,3,1,2]).to(device)
        if exploit:
            a=torch.argmax(self.forward(state))
        else:
            with torch.no_grad():
                q = self.forward(state)
                v = self.getV(q).squeeze()
                dist = torch.exp((q-v)/self.alpha)
                dist = dist / torch.sum(dist)
                c = Categorical(dist)
                a = c.sample()
        
        return a.item()

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
    
