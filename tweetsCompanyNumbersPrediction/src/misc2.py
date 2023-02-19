'''
Created on 15.02.2023

@author: vital
'''

from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper

modelpath =  DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModel"
topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
topic_words, word_scores, topic_nums = topicExtractor.get_topics()
print(topic_words[0])


