import torch
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.Dataloader import Dataloader
from classifier.FNN import FNN
from torch import nn
from classifier.Trainer import Trainer

featuresDf = pd.read_pickle(DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAmazon.pkl')
#featuresDf = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')
x = featuresDf.iloc[:,1:301].to_numpy()
y = featuresDf["class"].to_numpy()

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

deviceToUse = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader =  Dataloader(x,y,deviceToUse=deviceToUse ).getTrainsetDataloader()   
    
model = FNN(input_shape=x.shape[1],deviceToUse=deviceToUse).to(deviceToUse)
trainer = Trainer(loss_function = nn.BCELoss(), optimizer = torch.optim.SGD(model.parameters(),lr=0.01),epochs = 10, deviceToUse=deviceToUse)
trainer.train(model, x, y, trainloader)