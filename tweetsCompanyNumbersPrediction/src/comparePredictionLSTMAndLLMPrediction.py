'''
Created on 27.01.2024

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics

tweetGroupDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5_first_N.csv")
true_classes = tweetGroupDf["class"].tolist()
llm_prediction_classes = tweetGroupDf["chatgpt_class"].tolist()
metrics = BinaryClassificationMetrics() 
print(metrics.classification_report(true_classes, llm_prediction_classes ))