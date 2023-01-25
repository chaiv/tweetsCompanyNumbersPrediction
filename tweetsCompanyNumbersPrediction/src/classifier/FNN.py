'''
Created on 25.01.2023

@author: vital
'''
import torch
from torch import nn

class FNN(nn.Module):
    def __init__(self,input_shape,deviceToUse =torch.device("cuda:0")):
        super(FNN,self).__init__()
        self.fc1 = nn.Linear(input_shape,32,device=deviceToUse)
        self.fc2 = nn.Linear(32,64,device=deviceToUse)
        self.fc3 = nn.Linear(64,1,device=deviceToUse)  
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x