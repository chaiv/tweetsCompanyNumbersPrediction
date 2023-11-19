'''
Created on 19.11.2023

@author: vital
'''
import json 
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df.fillna('', inplace=True) #nan values in body columns
tweetSentences = df["body"].tolist()
lookupDict = TokenizedWordsLookup().createLookUpDictionary(tweetSentences)
with open(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tokenizerLookupAmazon.csv", 'w') as f:
    f.write(json.dumps(lookupDict))
