'''
Created on 19.11.2023

@author: vital
'''
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
tweetSentences = df["body"].tolist()
lookUpDict = {}
