'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from PredictionModelPath import MICROSOFT_GROSS_PROFIT_20,\
    MICROSOFT_XBOX_USERS_20


predictionModelPath = MICROSOFT_XBOX_USERS_20


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\CompanyTweets.csv")
numbers = pd.read_csv (predictionModelPath.getFinancialNumbersPath())
textfiltetedTweetsWithNumbers = FeatureDataframePipeline().createTweetWithNumbersDf(tweets,numbers,TweetQueryParams(companyName=predictionModelPath.getNasdaqTag()))
print(textfiltetedTweetsWithNumbers)
textfiltetedTweetsWithNumbers.to_csv(predictionModelPath.getDataframePath())

