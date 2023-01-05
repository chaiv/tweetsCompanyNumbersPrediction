'''
Created on 22.01.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector



class TweetsNumbersConnectorTest(unittest.TestCase):
    
    def testWhenPostDateProvidedOk(self):   
        figures =  pd.DataFrame(
                  [
                  (1,5,100),
                  (5,6,101)
                  ],
                  columns=["from_tsp","to_tsp","value"]
                  )
        connector = TweetNumbersConnector()
        self.assertEqual(100, connector.getFiguresValue(figures, 2,2) )
        self.assertEqual(101, connector.getFiguresValue(figures, 6,6) )
        self.assertEqual(100, connector.getFiguresValue(figures, 2) )
        self.assertEqual(101, connector.getFiguresValue(figures, 6) )
    
    def testWhenSameForMultipleNumbersRowThenFails(self):        
        figures =  pd.DataFrame(
                  [
                  (1,5,100),
                  (5,6,101)
                  ],
                  columns=["from_tsp","to_tsp","value"]
                  )
        connector = TweetNumbersConnector()
        with self.assertRaises(Exception):
            connector.getFiguresValue(figures, 8)  
        with self.assertRaises(Exception):
            connector.getFiguresValue(figures, 5)      
        pass


    def testTweetNumbersConnector(self):
        
        
        tweets = pd.DataFrame(
                  [
                  (1483230660),
                  (1451607900),
                  (1420070457),
                  (1483230660),
                  (1420070457)
                  ],
                  columns=["post_tsp"]
                  )
        
        figures =  pd.DataFrame(
                  [
                  (1420066800,1427752800,939880000.15),
                  (1451602800,1459375200,1200000000.15),
                  (1483225200,1490911200,963800000.15)
                  ],
                  columns=["from_tsp","to_tsp","value"]
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