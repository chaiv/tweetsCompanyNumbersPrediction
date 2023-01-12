'''
Created on 04.12.2022

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicModelCreator import TopicModelCreator

class Test(unittest.TestCase):


    def testModelCreateAndSave(self):
        modelPath = r'G:\Meine Ablage\promotion\companyTweets\TopicModelAAPLFirst1000'
        tweets = pd.read_csv (r'G:\Meine Ablage\promotion\companyTweets\CompanyTweetsAAPLFirst1000.csv')
        model = TopicModelCreator(1).createModel(tweets["body"].astype("string").tolist(), tweets["tweet_id"].tolist())
        model.save( modelPath)
        self.assertIsNotNone(TopicModelCreator(1).load( modelPath)) 
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testModelCreateAndSave']
    unittest.main()