'''
Created on 08.02.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from exploredata.TweetDataframeExplore import TweetDataframeExplore
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
explore= TweetDataframeExplore(df)
df.fillna('', inplace=True) #nan values in body columns
print(explore.getClassDistribution())
print(explore.getMostFrequentWords(100))