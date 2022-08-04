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
        tweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsFirst1000.csv')
        topicExtractor = TopicExtractor(TopicModelCreator(1).createModel(tweets["body"].tolist()))
        print(topicExtractor.getNumTopics())
        topic_words, word_scores, topic_nums =topicExtractor.get_topics()
        print(topic_words)
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()
