'''
Created on 01.03.2023

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicHeader import AbstractTopicHeaderFinder
from topicmodelling.TopicHeaderAddToDataframe import TopicHeaderAddToDataframe

class MockTopicHeaderFinder(AbstractTopicHeaderFinder):
    def getTopicHeader(self, word):
        if word == "cat":
            return (1, "animals")
        elif word == "apple":
            return (2, "fruits")
        else:
            return None
        
        
class Test(unittest.TestCase):


    def test_addTopicHeadersToWordsDataframe(self):
        data = {
            "token": ["cat","apple"]
        }
        df = pd.DataFrame(data)
        topicHeaderFinder = MockTopicHeaderFinder()
        topicHeaderAdder = TopicHeaderAddToDataframe(topicHeaderFinder)
        topicHeaderAdder.addTopicHeadersToWordsDataframe(df)
        self.assertListEqual(list(df.columns), ["token", "topicId","topicHeader"])
        self.assertListEqual(list(df["topicHeader"]), ["animals","fruits"])
        self.assertListEqual(list(df["topicId"]), [1,2])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()