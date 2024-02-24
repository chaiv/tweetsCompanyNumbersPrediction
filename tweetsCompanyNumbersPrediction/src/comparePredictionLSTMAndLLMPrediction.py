'''
Created on 27.01.2024

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics

tweetGroupDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5_first_N.csv")
true_classes = tweetGroupDf["class"].tolist()

with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5_predicted_chatgpt.txt", 'r') as file:
    content = ''.join(line.strip() for line in file)
llm_prediction_classes = [float(num) for num in content.split(',') if num.strip()]
metrics = BinaryClassificationMetrics() 
print(metrics.classification_report(true_classes, llm_prediction_classes ))
print(metrics.calculate_mcc(true_classes, llm_prediction_classes ))