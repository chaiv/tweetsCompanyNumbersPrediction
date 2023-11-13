'''
Created on 12.11.2023

@author: vital
'''

import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
importantWordsDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1Amazon.csv")
tweetIdColumnName = "tweet_id"
tokenColumnName = "token"
tokenAttributionColumnName = "token_attribution"
tweetAttributionColumnName = "tweet_attribution"
importantWordsDfSortedByAttributionAsc = importantWordsDf.sort_values(by=[tokenAttributionColumnName,tweetAttributionColumnName], ascending=True)
print(importantWordsDfSortedByAttributionAsc[[tweetIdColumnName,tokenColumnName,tokenAttributionColumnName,tweetAttributionColumnName]].head(100))