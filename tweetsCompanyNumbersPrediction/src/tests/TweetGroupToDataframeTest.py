'''
Created on 14.01.2024

@author: vital
'''
import unittest
from nlpvectors.TweetGroup import TweetGroup
from nlpvectors.TweetGroupToDataframe import TweetGroupToDataframe


class TweetGroupToDataframeTest(unittest.TestCase):


    def testTweetGroupToDataframe(self):
        tweetGroup1 = TweetGroup(sentences=["tweet 1", "second tweet"],sentenceIds=[123,456],totalTokenIndexes=[[1,2],[1,2]],
                                 totalTokens=[["tweet", "1"], ["second", "tweet"]],
                                 totalFeatureVector=[1,2,0,1,2],separatorIndexesInFeatureVector=[2],label = 0)
        tweetGroup2 = TweetGroup(sentences=["next tweet", "tweet 4"],sentenceIds=[789,101112],totalTokenIndexes=[[1,2],[1,2]],
                                 totalTokens=[["next", "tweet"], ["tweet", "4"]],
                                 totalFeatureVector=[3,1,0,1,5],separatorIndexesInFeatureVector=[2],label = 1)
        
        tweetGroupToDataframe = TweetGroupToDataframe()
        
        df = tweetGroupToDataframe.createTweetGroupDataframe([tweetGroup1, tweetGroup2])
        
        self.assertEquals(2,len(df))
        self.assertEquals(df["tweet_sentences"].iloc[0],"tweet 1;second tweet")
        self.assertEquals(df["label"].iloc[0],0)
        self.assertEquals(df["tweet_sentences"].iloc[1],"next tweet;tweet 4")
        self.assertEquals(df["label"].iloc[1],1)



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()