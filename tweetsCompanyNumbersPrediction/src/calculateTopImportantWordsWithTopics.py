'''
Created on 12.11.2023

@author: vital
'''

import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
importantWordsDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\importantWordsClass1Amazon.csv")
sentenceIdColumnName = "id"
tokenColumnName = "token"
attributionColumnName = "attribution"
importantWordsDfSortedByAttributionAsc = importantWordsDf.sort_values(by=attributionColumnName, ascending=True)
print(importantWordsDfSortedByAttributionAsc[[sentenceIdColumnName,tokenColumnName,attributionColumnName]].head(50))