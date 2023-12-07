'''
Created on 07.12.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper

tweetDf = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
randomNSamplesDf = tweetDf.sample(n=1000)
randomNSamplesDf[["tweet_id","body"]].to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\random1000SamplesAmazon.csv")

