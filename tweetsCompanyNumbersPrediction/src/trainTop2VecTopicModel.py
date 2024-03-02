'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import Top2VecTopicModelCreator
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from PredictionModelPath import MICROSOFT_EPS_5

predictionModelPath = MICROSOFT_EPS_5 

#tweetsFile = "CompanyTweetsAAPLFirst1000.csv"
#topicModelFile = "TopicModelAAPLFirst1000V2"

#tweetsFile = "amazonTweetsWithNumbers.csv"
#topicModelFile = "amazonTopicModelV2"

#tweetsFile = "amazonTweetsWithNumbers.csv"
#topicModelFile = "amazonTopicModelRandom15000"


tweets = pd.read_csv (MICROSOFT_EPS_5.getDataframePath())
tweets.fillna('', inplace=True) #nan values in body columns 
tokenizer = TweetTokenizer(DefaultWordFilter())
documents = []


for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = Top2VecTopicModelCreator().createModel(documents, tweets["tweet_id"].tolist(),tokenizer= tokenizer.tokenize)
model.save(MICROSOFT_EPS_5.getTop2VecModelPath())

