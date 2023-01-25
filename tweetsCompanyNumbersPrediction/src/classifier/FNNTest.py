#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

#importing the dataset
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.Dataloader import Dataloader

featuresDf = pd.read_pickle(DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAmazon.pkl')
#featuresDf = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')
x = featuresDf.iloc[:,1:301].to_numpy()
y = featuresDf["class"].to_numpy()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

deviceToUse = torch.device("cuda:0")
trainloader =  Dataloader(x,y).getTrainsetDataloader()   
    
    
#TODO Testset loader
from torch import nn
from torch.nn import functional as F
class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,32,device=deviceToUse)
        self.fc2 = nn.Linear(32,64,device=deviceToUse)
        self.fc3 = nn.Linear(64,1,device=deviceToUse)  
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
#hyper parameters
learning_rate = 0.01
epochs = 700# Model , Optimizer, Loss
model = Net(input_shape=x.shape[1]).to(deviceToUse)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
loss_fn = nn.BCELoss()

#forward loop
losses = []
accur = []
for i in range(epochs):
    for j,(x_train,y_train) in enumerate(trainloader):
    
    #calculate output
        output = model(x_train.to(deviceToUse ))
 
    #calculate loss
        loss = loss_fn(output,y_train.reshape(-1,1))
 
    #accuracy
        predicted = model(torch.tensor(x,dtype=torch.float32).to(deviceToUse ))
        acc = (predicted.reshape(-1).detach().cpu().numpy().round() == y).mean()    #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
        accur.append(acc)
        print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))