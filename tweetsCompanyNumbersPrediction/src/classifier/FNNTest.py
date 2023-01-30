import torch
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.Dataloader import Dataloader
from classifier.FNN import FNN
from torch import nn
from classifier.Trainer import Trainer

#https://medium.com/@CVxTz/add-interpretability-to-your-nlp-model-the-easy-way-using-captum-ec56f538f746
#https://stats.stackexchange.com/questions/258166/good-accuracy-despite-high-loss-value

featuresDf = pd.read_pickle(DataDirHelper().getDataDir()+ 'companyTweets\\featuresClassesAmazon.pkl')
#featuresDf = pd.read_csv(DataDirHelper().getDataDir() + 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')
train_endrow = int((len(featuresDf.index) * 0.7))
x = featuresDf.iloc[:, 1:301].to_numpy()
y = featuresDf["class"].to_numpy()

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

x_train = x[:train_endrow]
y_train = y[:train_endrow]

x_test = x[train_endrow:]
y_test = y[train_endrow:]

deviceToUse = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader = Dataloader(x, y, deviceToUse=deviceToUse).getTrainsetDataloader()   
    
model = FNN(input_shape=x.shape[1], deviceToUse=deviceToUse).to(deviceToUse)
trainer = Trainer(
    loss_function=nn.BCELoss(),
    optimizer=torch.optim.SGD(model.parameters(), lr=0.01), 
    model=model, 
    epochs=50, 
    deviceToUse=deviceToUse
    )
trainer.train(x_test, y_test, trainloader)
