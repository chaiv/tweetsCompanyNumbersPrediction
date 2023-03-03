'''
Created on 01.03.2023

@author: vital
'''
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder

class TopicHeaderAddToDataframe(object):

    def __init__(self, topicHeaderFinder: AbstractTopicHeaderFinder, tweetIdColumnName = "tweet_id",topicIdColumnName = "topicId",topicHeaderColumnName = "topicHeader"):
        self.topicHeaderFinder = topicHeaderFinder
        self.tweetIdColumnName = tweetIdColumnName
        self.topicIdColumnName = topicIdColumnName
        self.topicHeaderColumnName = topicHeaderColumnName
     
     
    def addTopicHeadersToWordsDataframe(self, wordsDf):   
        topic_nums, topic_headers = self.topicHeaderFinder.getTopicHeaderByIds(wordsDf[self.tweetIdColumnName].tolist())
        wordsDf[self.topicIdColumnName] = topic_nums
        wordsDf[self.topicHeaderColumnName] = topic_headers

        