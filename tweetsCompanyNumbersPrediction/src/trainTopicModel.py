'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonTweetsWithNumbers.csv")  
tokenizer = TweetTokenizer(DefaultWordFilter())
documents = []
for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = TopicModelCreator().createModel(documents, tweets["tweet_id"].tolist(),tokenizer= tokenizer.tokenize)
model.save(DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelV2")

