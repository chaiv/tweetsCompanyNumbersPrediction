'''
Created on 18.02.2023

@author: vital
'''
import numpy as np
import unittest
from topicmodelling.TopicHeader import TopicHeaderCalculator
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from tweetpreprocess.DataDirHelper import DataDirHelper


class TopicHeaderFinderTestCase(unittest.TestCase):
    def setUp(self):
        self.topic_words = ["apple", "banana", "orange", "kiwi", "grape"]
        self.topic_word_vectors = [
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10]),
            np.array([11, 12, 13, 14, 15]),
            np.array([16, 17, 18, 19, 20]),
            np.array([21, 22, 23, 24, 25])
        ]
        self.topic_header_finder = TopicHeaderCalculator()

    def test_get_header(self):
        # Test that the function returns the expected header
        expected_header = "orange"
        header = self.topic_header_finder.calculateHeader(self.topic_words, self.topic_word_vectors)
        self.assertEqual(header, expected_header)

        # Test that the function returns None if given an empty list of words
        header = self.topic_header_finder.calculateHeader([], [])
        self.assertIsNone(header)

        # Test that the function returns None if given an empty list of word vectors
        header = self.topic_header_finder.calculateHeader(self.topic_words, [])
        self.assertIsNone(header)
        
    def test_with_topic_model(self):
        modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000"
        topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))    
        topic_words, word_scores, topic_nums = topicExtractor.get_topics()
        observedTopicWords = topic_words[0]
        self.assertEqual("writer",self.topic_header_finder.calculateHeader(observedTopicWords, topicExtractor.getWordVectorsOfWords(topic_words[0])))
        self.assertEqual("writer", topicExtractor.getTopicHeader("financialnews")[1]);