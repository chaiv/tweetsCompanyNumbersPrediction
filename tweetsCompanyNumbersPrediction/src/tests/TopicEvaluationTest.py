'''
Created on 17.04.2023

@author: vital
'''
import unittest
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicModelCreator import TopicModelCreator
from topicmodelling.TopicEvaluation import TopicEvaluation
from topicmodelling.TopicExtractor import TopicExtractor



class TopicEvaluationTest(unittest.TestCase):
    
    modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000"
    topicEvaluation = TopicEvaluation(TopicModelCreator().load(modelpath))

    def testCosineSim(self):
        print(self.topicEvaluation.getCosineSimilarityMatrix())

    def testCoherence(self):
        #print(self.topicEvaluation.get_topic_coherence())
        pass
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()