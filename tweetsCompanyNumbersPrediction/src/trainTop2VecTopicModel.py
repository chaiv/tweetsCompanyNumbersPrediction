'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from PredictionModelPath import MICROSOFT_EPS_5, GOOGLE_SE_MARKET_SHARE_10

predictionModelPath = GOOGLE_SE_MARKET_SHARE_10 

tweets = pd.read_csv (predictionModelPath.getDataframePath())
tweets.fillna('', inplace=True) #nan values in body columns 
tokenizer = TweetTokenizer(DefaultWordFilter())
documents = []


for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = Top2VecTopicModelCreator().createModel(documents, tweets["tweet_id"].tolist(),tokenizer= tokenizer.tokenize)
model.save(predictionModelPath.getTop2VecModelPath())

