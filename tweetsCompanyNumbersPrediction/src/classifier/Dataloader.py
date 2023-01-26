'''
Created on 25.01.2023

@author: vital
'''
from torch.utils.data import Dataset, DataLoader
import torch

class Dataset(Dataset):
    def __init__(self,x,y,deviceToUse):
        self.x = torch.tensor(x,dtype=torch.float32).to(deviceToUse )
        self.y = torch.tensor(y,dtype=torch.float32).to(deviceToUse )
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    def __len__(self):
        return self.length

class Dataloader(object):

    def __init__(self,xTrain,yTrain,xTest=None,yTest=None, batch_size=258944 ,shuffle=False,deviceToUse=torch.device("cuda:0")):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.deviceToUse = deviceToUse
    
    def getTrainsetDataloader(self):
        return DataLoader(Dataset(self.xTrain,self.yTrain,self.deviceToUse),self.batch_size ,self.shuffle)
    
        