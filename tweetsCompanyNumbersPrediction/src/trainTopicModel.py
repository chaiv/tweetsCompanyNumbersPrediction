'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonTweetsWithNumbers.csv")  
documents = []
for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = TopicModelCreator().createModel(documents, tweets["tweet_id"].tolist())
model.save(DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModel")

