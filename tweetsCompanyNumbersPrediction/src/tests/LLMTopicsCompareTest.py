'''
Created on 10.12.2023

@author: vital
'''
import unittest
import pandas as pd
import numpy as np 
from topicmodelling.llmcomparison.LLMTopicsCompare import LLMTopicsCompare
from topicmodelling.TopicExtractor import Top2VecTopicExtractor
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter

class TopicExtractorFake(Top2VecTopicExtractor):
    
    def __init__(self):
        pass
    
    def get_documents_topics(self,tweetIds,num_topics):
        return [[1,2],[1],[2],[]],None,None,None
    
    def searchTopics(self,keywords, num_topics):
        topic_nums =[]
        if(keywords == ["sharehold"]):
            topic_nums=np.array([1,5]) 
        if(keywords == ["stock"]):
            topic_nums=np.array([1])   
        if(keywords == ["new"]):
            topic_nums=np.array([2,3])       
        if(keywords == ["ticker"]):
            topic_nums=np.array([1])       
        if(keywords == ["oil"]):    
            topic_nums = [4]        
        return None,None,None,topic_nums 


class LLMTopicsCompareTest(unittest.TestCase):


    def testcalculatePercentageOfTrue(self):
        topicsCompare = LLMTopicsCompare(None,None,None,None)
        self.assertEqual(0.5,topicsCompare.calculatePercentageOfTrue([[True,False],[False,True]]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([[False,False],[False,False]]))
        self.assertEqual(1,topicsCompare.calculatePercentageOfTrue([[True,True],[True,True]]))
        self.assertEqual(0.25,topicsCompare.calculatePercentageOfTrue([[True,False],[False,False],[]]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([]))
        self.assertEqual(0,topicsCompare.calculatePercentageOfTrue([[]]))
        
    def testMostSimilarEmbeddingIndexes(self):
        topicsCompare = LLMTopicsCompare(None,None,None,None)
        allEmbeddings = np.array([
            [0,1,2],
            [0,0,0],
            [1,1,1],
            [4,0,1],
            [1,1,2]
            ])
        mostSimilarEmbeddingIndexes1 = topicsCompare.getMostSimilarEmbeddingIndexes(allEmbeddings, np.array([[1,1,1]]), 3)
        self.assertEqual(2,mostSimilarEmbeddingIndexes1[0])
        self.assertEqual(4,mostSimilarEmbeddingIndexes1[1])
        self.assertEqual(0,mostSimilarEmbeddingIndexes1[2])
        mostSimilarEmbeddingIndexes2 = topicsCompare.getMostSimilarEmbeddingIndexes(allEmbeddings,  np.array([[4,0,0]]), 1)
        self.assertEqual(3,mostSimilarEmbeddingIndexes2[0])
        

    
        
        



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()