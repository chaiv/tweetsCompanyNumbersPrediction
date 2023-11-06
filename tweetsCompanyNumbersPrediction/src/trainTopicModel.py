'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer



#tweetsFile = "CompanyTweetsAAPLFirst1000.csv"
#topicModelFile = "TopicModelAAPLFirst1000V2"

#tweetsFile = "amazonTweetsWithNumbers.csv"
#topicModelFile = "amazonTopicModelV2"

tweetsFile = "CompanyTweetsTeslaWithCarSales.csv"
topicModelFile = "teslaTopicModel"
tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\"+tweetsFile)  
tokenizer = TweetTokenizer(DefaultWordFilter())
documents = []
for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = TopicModelCreator().createModel(documents, tweets["tweet_id"].tolist(),tokenizer= tokenizer.tokenize)
model.save(DataDirHelper().getDataDir()+ "companyTweets\\"+topicModelFile)

