'''
Created on 27.01.2024

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from PredictionModelPath import AMAZON_20

predictionModelPath = AMAZON_20


tweetGroupDf = pd.read_csv(predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+".csv")
firstNZeroClasses = tweetGroupDf[tweetGroupDf["class"]==0.0].head(50)
firstNOneClasses = tweetGroupDf[tweetGroupDf["class"]==1.0].head(50)
firstNTweetGroupsDfRandomOrder = pd.concat([firstNZeroClasses, firstNOneClasses], ignore_index=True).sample(frac=1)
firstNTweetGroupsDfRandomOrder["tweet_sentences"] = firstNTweetGroupsDfRandomOrder["tweet_sentences"].apply(lambda x: str(x) + "<SEP>")
firstNTweetGroupsDfRandomOrder.to_csv(predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+"_first_N.csv")

