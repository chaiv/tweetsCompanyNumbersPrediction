'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from PredictionModelPath import  AMAZON_REVENUE_10_LSTM_MULTI_CLASS,\
    APPLE__EPS_10_LSTM_MULTI_CLASS, TESLA_CAR_SALES_10_LSTM_MULTI_CLASS
from tweetpreprocess.FiguresMultiClassCalculator import FiguresMultiClassCalculator


predictionModelPath = TESLA_CAR_SALES_10_LSTM_MULTI_CLASS


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\CompanyTweets.csv")
numbers = pd.read_csv (predictionModelPath.getFinancialNumbersPath())
classCalculator = FiguresMultiClassCalculator()
textfiltetedTweetsWithNumbers = FeatureDataframePipeline().createTweetWithNumbersDf(tweets,numbers,TweetQueryParams(companyNames=predictionModelPath.getNasdaqTag()),classCalculator)
print(textfiltetedTweetsWithNumbers)
textfiltetedTweetsWithNumbers.to_csv(predictionModelPath.getDataframePath())

