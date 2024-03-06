'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from PredictionModelPath import MICROSOFT_GROSS_PROFIT_20,\
    MICROSOFT_XBOX_USERS_20, APPLE_IPHONE_SALES_10, GOOGLE_SE_MARKET_SHARE_10


predictionModelPath = GOOGLE_SE_MARKET_SHARE_10


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\CompanyTweets.csv")
numbers = pd.read_csv (predictionModelPath.getFinancialNumbersPath())
textfiltetedTweetsWithNumbers = FeatureDataframePipeline().createTweetWithNumbersDf(tweets,numbers,TweetQueryParams(companyNames=predictionModelPath.getNasdaqTag()))
print(textfiltetedTweetsWithNumbers)
textfiltetedTweetsWithNumbers.to_csv(predictionModelPath.getDataframePath())

