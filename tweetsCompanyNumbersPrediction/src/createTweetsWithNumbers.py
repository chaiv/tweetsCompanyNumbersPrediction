'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from pipeline.FeatureDataframePipeline import FeatureDataframePipeline
from tweetpreprocess.TweetQueryParams import TweetQueryParams

tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\CompanyTweets.csv")
numbers = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\microsoftEPS.csv")
textfiltetedTweetsWithNumbers = FeatureDataframePipeline().createTweetWithNumbersDf(tweets,numbers,TweetQueryParams(companyName="MSFT"))
print(textfiltetedTweetsWithNumbers)
textfiltetedTweetsWithNumbers.to_csv(DataDirHelper().getDataDir()+"companyTweets\CompanyTweetsMicrosoftWithEps.csv")

