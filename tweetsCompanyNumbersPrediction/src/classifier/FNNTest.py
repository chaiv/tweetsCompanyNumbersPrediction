import torch
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.Dataloader import Dataloader
from classifier.FNN import FNN
from torch import nn

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
    
#hyper parameters
learning_rate = 0.01
epochs = 700# Model , Optimizer, Loss
model = FNN(input_shape=x.shape[1]).to(deviceToUse)
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