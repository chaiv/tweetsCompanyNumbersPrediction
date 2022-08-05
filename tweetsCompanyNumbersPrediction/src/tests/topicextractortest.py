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
        tweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsAppleFirst1000.csv')
        topicExtractor = TopicExtractor(TopicModelCreator(1).createModel(tweets["body"].tolist()))
        self.assertTrue(topicExtractor.getNumTopics()>0)
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()
