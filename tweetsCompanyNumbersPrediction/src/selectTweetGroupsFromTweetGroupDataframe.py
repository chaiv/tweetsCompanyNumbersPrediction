'''
Created on 27.01.2024

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper

tweetDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5.csv")
firstNZeroClasses = tweetDf[tweetDf["class"]==0.0].head(25)
firstNOneClasses = tweetDf[tweetDf["class"]==1.0].head(25)
firstNTweetGroupsDfRandomOrder = pd.concat([firstNZeroClasses, firstNOneClasses], ignore_index=True).sample(frac=1)
firstNTweetGroupsDfRandomOrder.to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5_first_N.csv")

