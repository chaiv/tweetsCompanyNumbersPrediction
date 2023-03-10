'''
Created on 10.03.2023

@author: vital
'''
import json 
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.VocabularyCreator import VocabularyCreator
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
tweets = pd.read_csv (DataDirHelper().getDataDir()+"companyTweets\CompanyTweetsTeslaWithCarSales.csv")
dictionaryPath = DataDirHelper().getDataDir()+ "companyTweets\VocabularyTesla.json"
tweets.fillna('', inplace=True) #nan values in body columns
sentences = tweets["body"].tolist()
with open(dictionaryPath, 'w') as f:
    f.write(json.dumps(VocabularyCreator(TweetTokenizer(DefaultWordFilter())).createVocabularyDict(sentences)))

