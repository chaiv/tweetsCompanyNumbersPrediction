'''
Created on 29.01.2022

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator


class TopicExtractorTest(unittest.TestCase):

    def testTopicExtraction(self):
        modelpath =  r'G:\Meine Ablage\promotion\companyTweets\TopicModelAAPLFirst100000.csv'
        topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
        self.assertTrue(topicExtractor.getNumTopics()>0)
        print(topicExtractor.get_topicwords())
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()
