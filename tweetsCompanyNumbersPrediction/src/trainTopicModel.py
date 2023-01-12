'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator

tweets = pd.read_csv (r'C:\Users\vital\Google Drive\promotion\companyTweets\amazonTweetsWithNumbers')  
documents = []
for index, row in tweets.iterrows():
    documents.append(str(row["body"]))
model = TopicModelCreator(5).createModel(documents, tweets["tweet_id"].tolist())
model.save(r"C:\Users\vital\Desktop\df\amazonTopicModel")

