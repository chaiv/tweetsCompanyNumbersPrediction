'''
Created on 17.04.2023

@author: vital
'''
import unittest
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicEvaluation import TopicEvaluation
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from topicmodelling.TopicExtractor import TopicExtractor



class TopicEvaluationTest(unittest.TestCase):
    
    modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000V2"
    topicEvaluation = TopicEvaluation(TopicExtractor(TopicModelCreator().load(modelpath)),TweetTokenizer(DefaultWordFilter()))

    def testCosineSim(self):
        self.assertEqual(
            [ 1.000000238418579,0.3766742944717407 ,-0.9657979607582092,-0.15776774287223816 ,-0.8605716228485107 ,-0.4055122137069702]
            ,self.topicEvaluation.getCosineSimilarityMatrix()[0].tolist())

    def testCoherence(self):
        self.assertEqual(0.7406706138618926,self.topicEvaluation.get_topic_coherence())

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()