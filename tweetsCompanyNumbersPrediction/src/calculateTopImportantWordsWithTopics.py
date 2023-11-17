'''
Created on 12.11.2023

@author: vital
'''

import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from featureinterpretation.ImportantWordsDataframeUtil import addUntokenizedWordColumnFromTweetDf,\
    addPOSTagsColumn
from exploredata.POSTagging import PartOfSpeechTagging
tweetDf = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
importantWordsDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1Amazon.csv")
tweetIdColumnName = "tweet_id"
bodyColumnName = "body"
originaltokenColumnName = "original_token"
tokenAttributionColumnName = "token_attribution"
tweetAttributionColumnName = "tweet_attribution"
tweetPos = "tweet_pos"
posTagColumnName = "token_pos"
importantWordsDf = importantWordsDf.sort_values(by=[tokenAttributionColumnName,tweetAttributionColumnName], ascending=False).head(100)
importantWordsDf = addUntokenizedWordColumnFromTweetDf(tweetDf,importantWordsDf)
importantWordsDf = addPOSTagsColumn(PartOfSpeechTagging(),importantWordsDf)
importantWordsDf[[originaltokenColumnName,posTagColumnName,bodyColumnName,tweetPos,tokenAttributionColumnName,tweetAttributionColumnName]].to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1AmazonSorted.csv")