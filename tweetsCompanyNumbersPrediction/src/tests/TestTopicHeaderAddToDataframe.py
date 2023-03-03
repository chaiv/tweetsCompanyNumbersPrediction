'''
Created on 01.03.2023

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder
from topicmodelling.TopicHeaderAddToDataframe import TopicHeaderAddToDataframe

class MockTopicHeaderFinder(AbstractTopicHeaderFinder):
    def getTopicHeaderByIds(self,documentIds):
        return [0,1],["topic1","topic2"]
        
        
class Test(unittest.TestCase):


    def test_addTopicHeadersToWordsDataframe(self):
        data = {
            "tweet_id": ["id1","id2"]
        }
        df = pd.DataFrame(data)
        topicHeaderFinder = MockTopicHeaderFinder()
        topicHeaderAdder = TopicHeaderAddToDataframe(topicHeaderFinder)
        topicHeaderAdder.addTopicHeadersToWordsDataframe(df)
        self.assertListEqual(list(df.columns), ["tweet_id", "topicId","topicHeader"])
        self.assertListEqual(list(df["topicHeader"]), ["topic1","topic2"])
        self.assertListEqual(list(df["topicId"]), [0,1])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()