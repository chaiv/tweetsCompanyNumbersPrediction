'''
Created on 27.01.2024

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics
from PredictionModelPath import AMAZON_20


predictionModelPath = AMAZON_20

tweetGroupDf = pd.read_csv(
    predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+"_first_N.csv")
true_classes = tweetGroupDf["class"].tolist()
with open(predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+"_predicted_chatgpt.txt", 'r') as file:
    content = ''.join(line.strip() for line in file)
llm_prediction_classes = [float(num) for num in content.split(',') if num.strip()]
metrics = BinaryClassificationMetrics() 
print(metrics.classification_report(true_classes, llm_prediction_classes ))
print(metrics.calculate_mcc(true_classes, llm_prediction_classes ))