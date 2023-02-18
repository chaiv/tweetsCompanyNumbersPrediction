'''
Created on 15.02.2023

@author: vital
'''

from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper

modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\TopicModelAAPLFirst1000"
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
print(topicExtractor.searchTopics(["cheating"], 2))


