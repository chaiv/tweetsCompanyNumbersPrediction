'''
Created on 10.01.2023

@author: vital
'''
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator

tweets = pd.read_csv (r"C:\Users\vital\Desktop\df\amazonTweetsWithNumbers")  
model = TopicModelCreator(5).createModel(tweets["body"].tolist(), tweets["tweet_id"].tolist())
model.save(r"C:\Users\vital\Desktop\df\amazonTopicModel")

