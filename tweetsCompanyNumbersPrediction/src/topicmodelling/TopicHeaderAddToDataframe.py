'''
Created on 01.03.2023

@author: vital
'''
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder

class TopicHeaderAddToDataframe(object):

    def __init__(self, topicHeaderFinder: AbstractTopicHeaderFinder, wordColumnName = "token",topicIdColumnName = "topicId",topicHeaderColumnName = "topicHeader"):
        self.topicHeaderFinder = topicHeaderFinder
        self.wordColumnName = wordColumnName
        self.topicIdColumnName = topicIdColumnName
        self.topicHeaderColumnName = topicHeaderColumnName
     
     
    def addTopicHeadersToWordsDataframe(self, wordsDf):    
        wordsDf[self.topicIdColumnName], wordsDf[self.topicHeaderColumnName] = zip(*wordsDf[self.wordColumnName].apply(self.topicHeaderFinder.getTopicHeader))
        