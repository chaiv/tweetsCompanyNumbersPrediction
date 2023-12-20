'''
Created on 20.12.2023

@author: vital
'''
import pandas as pd
import unittest
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import BertTopicExtractor
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer


class TopicExtractorBertTest(unittest.TestCase):


    
    def setUp(self):
        self.topicExtractor = BertTopicExtractor(
            DataDirHelper().getDataDir()+ "companyTweets\\amazonTopicModelBertFirst1000",
            TweetTokenizer(DefaultWordFilter()),
            pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\amazonBertTopicMappingFirst1000.csv")
            )
        
        
    def testGetNumberOfTopics(self):
        self.assertEquals(29, self.topicExtractor.getNumberOfTopics())
        
    def testGetTopicWordsScoresAndIds(self):
        topic_words,word_scores,topic_ids = self.topicExtractor.getTopicWordsScoresAndIds()
        self.assertEquals(30,len(topic_ids))
        self.assertEquals('servic',topic_words[0][0])

    def testGetDocumentTopicWordsTopicScoresAndTopicIds(self):
        topic_words,word_scores,topic_ids = self.topicExtractor.getDocumentTopicWordsTopicScoresAndTopicIds([550453624258965505],3)
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TopicExtractorBertTest.testName']
    unittest.main()