'''
Created on 19.01.2023

@author: vital
'''
import pandas as pd
from classifier.lstm.DatasetMapper import DatasetMaper
from torch.utils.data import DataLoader
from tweetpreprocess.DataDirHelper import DataDirHelper



featuresDf = pd.read_csv (DataDirHelper().getDataDir()+ 'companyTweets\\FeaturesClassesAAPLFirst1000.csv')

x_train = list(featuresDf.iloc[:,0:300].to_numpy())
y_train = list(featuresDf["class"])

print(x_train)
print(y_train)


x_test = list(featuresDf.iloc[:,0:300].to_numpy())
y_test = list(featuresDf["class"])



training_set = DatasetMaper(x_train, y_train)
test_set = DatasetMaper(x_test, y_test)
        
loader_training = DataLoader(training_set, batch_size=5)
loader_test = DataLoader(test_set)
