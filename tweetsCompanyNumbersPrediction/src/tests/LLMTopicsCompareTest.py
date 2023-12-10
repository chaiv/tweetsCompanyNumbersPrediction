'''
Created on 10.12.2023

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.llmcomparison.LLMTopicsCompare import LLMTopicsCompare
from topicmodelling.TopicExtractor import TopicExtractor

class TopicExtractorFake(TopicExtractor):
    
    def __init__(self):
        pass
    
    def get_documents_topics(self,tweetIds):
        return [[1,3],[1],[2],[]],None,None,None
    
    def searchTopics(self,keywords, num_topics):
        topic_nums =[]
        if(keywords == ["shareholder","stock"]):
            topic_nums=[1] #must be list of list because every topic word may have multiple topic nums
        if(keywords == ["news","ticker"]):    
            topic_nums = [2,3]
        if(keywords == ["oil"]):    
            topic_nums = [4]        
        return None,None,None,topic_nums 


class LLMTopicsCompareTest(unittest.TestCase):


    def testcalculatePercentageOfTrue(self):
        topicsCompare = LLMTopicsCompare(None,None)
        self.assertEqual(0.5,topicsCompare.calculatePercentageOfTrue([[True,False],[False,True]]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([[False,False],[False,False]]))
        self.assertEqual(1,topicsCompare.calculatePercentageOfTrue([[True,True],[True,True]]))
        self.assertEqual(0.25,topicsCompare.calculatePercentageOfTrue([[True,False],[False,False]]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([[]]))

    

    def testSimilarityScore(self):
        topics =  pd.DataFrame(
                [
                  (1,"shareholder;stock"),
                  (2,"news;ticker"),
                  (3,"oil"),
                  (4,"nothing"),
                  ],
                  columns=["tweet_id","topics"]
                )
        topicsCompare = LLMTopicsCompare(TopicExtractorFake(),topics)
        print(topicsCompare.calculateSimilarityScore("topics",2))
        
        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()