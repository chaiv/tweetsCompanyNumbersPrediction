'''
Created on 19.01.2023

@author: vital
'''
import pandas as pd
import torch
import torch.nn.functional as F
from classifier.lstm.DatasetMapper import DatasetMaper
from torch.utils.data import DataLoader
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.lstm.LSTM import LSTM
import torch.optim as optim


@staticmethod
def evaluation(self,model,loader_test):

    predictions = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
                
            y_pred = model(x)
            predictions += list(y_pred.detach().numpy())
                
    return predictions
            
@staticmethod
def calculate_accuray(grand_truth, predictions):
    true_positives = 0
    true_negatives = 0
        
    for true, pred in zip(grand_truth, predictions):
        if (pred > 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
                
    return (true_positives+true_negatives) / len(grand_truth)


featuresDf = pd.read_csv (DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')

x_train = list(featuresDf.iloc[:,0:300].to_numpy())
y_train = list(featuresDf["class"])

x_test = list(featuresDf.iloc[:,0:300].to_numpy())
y_test = list(featuresDf["class"])



training_set = DatasetMaper(x_train, y_train)
test_set = DatasetMaper(x_test, y_test)
        
loader_training = DataLoader(training_set, batch_size=5)
loader_test = DataLoader(test_set)


model = LSTM()

# Defines a RMSprop optimizer to update the parameters
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

epocs = 10

for epoch in range(epocs):
            
    predictions = []
            
    model.train()
            
    for x_batch, y_batch in loader_training:
                        
        x = x_batch.type(torch.FloatTensor)
        y = y_batch.type(torch.IntTensor)
                
        y_pred = model(x)
                
        loss = F.binary_cross_entropy(y_pred, y)
                
        optimizer.zero_grad()
                
        loss.backward()
                
        optimizer.step()
                
        predictions += list(y_pred.squeeze().detach().numpy())
            
        test_predictions = evaluation(model,loader_test)
            
        train_accuary = calculate_accuray(y_train, predictions)
        test_accuracy = calculate_accuray(y_test, test_predictions)
            
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))

