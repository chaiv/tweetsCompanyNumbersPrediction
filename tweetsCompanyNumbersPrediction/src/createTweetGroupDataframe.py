'''
Created on 15.01.2024

@author: vital
'''
from nlpvectors.TweetGroupToDataframe import TweetGroupToDataframe
from tweetpreprocess.DataDirHelper import DataDirHelper
import pandas as pd
from nlpvectors.DataframeSplitter import DataframeSplitter
import random

tweetDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv")
tweetDf.fillna('', inplace=True) #nan values in body columns 
splitter = DataframeSplitter()
splits = splitter.getSplitIds(tweetDf, 5) #how many tweets  as one sample
random_splits = random.sample(splits, 100)
for split in random_splits:
    splitDf =  tweetDf [ tweetDf ["tweet_id"].isin( split)]
    sentences = tweetDf ["body"].tolist()
    label = tweetDf["class"].iloc[0]

#tweetGroupDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups5.csv")
