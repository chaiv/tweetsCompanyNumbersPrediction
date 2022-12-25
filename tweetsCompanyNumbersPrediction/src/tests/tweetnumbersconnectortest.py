'''
Created on 22.01.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector



class TweetsNumbersConnectorTest(unittest.TestCase):


    def testTweetNumbersConnector(self):
        
        
        tweets = pd.DataFrame(
                  [
                  (1483230660),
                  (1451607900),
                  (1420070457),
                  (1483230660),
                  (1420070457)
                  ],
                  columns=["post_date"]
                  )
        
        figures =  pd.DataFrame(
                  [
                  (1420066800,1427752800,939880000.15),
                  (1451602800,1459375200,1200000000.15),
                  (1483225200,1490911200,963800000.15)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        
        
        connector = TweetNumbersConnector()
        allTweetsWithNumbersDf = connector.getTweetsWithNumbers(
            tweets, 
            figures
            )
        self.assertEqual( allTweetsWithNumbersDf.iloc[0]['value'],963800000.15)
        self.assertEqual( allTweetsWithNumbersDf.iloc[4]['value'],939880000.15)
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TweetsNumbersConnectorTest.testName']
    unittest.main()