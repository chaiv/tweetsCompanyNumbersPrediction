'''
Created on 18.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams

class TweetQueryTest(unittest.TestCase):
    
    
    def testWhenQueryParamIdsThenReturnSecondRow(self):
        tweetsDf = pd.DataFrame(
                  [
                  (1),
                  (2),
                  (3)
                  ],
                  columns=['tweet_id']
                  )
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams(tweetIds=[2]))
        self.assertEquals(1,resultDf.size)  
        self.assertEquals(2,resultDf.iloc[0]['tweet_id'])  
        


    def testWhenQueryNoParamTheReturnOriginalDf(self):
        tweetsDf = pd.DataFrame(
                  [
                  ("company1"),
                  ("company2"),
                  ("company3")
                  ],
                  columns=['ticker_symbol']
                  )
        
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams())
        self.assertTrue( tweetsDf.equals(resultDf))  
        pass
    
    def testWhenQueryParamCompanyTheReturnCompany1(self):
        tweetsDf = pd.DataFrame(
                  [
                  ("company1"),
                  ("company2"),
                  ("company3")
                  ],
                  columns=['ticker_symbol']
                  )
        
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams(companyName ="company1"))
        self.assertEquals(1,resultDf.size)  
        self.assertEquals("company1",resultDf.iloc[0]['ticker_symbol'])  
        pass
    
    def testWhenQueryParamNumberTheReturnTwoRows(self):
        tweetsDf = pd.DataFrame(
                  [
                  ("company1"),
                  ("company2"),
                  ("company3")
                  ],
                  columns=['ticker_symbol']
                  )
        
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams(firstNTweets =2))
        self.assertEquals(2,resultDf.size)  
        self.assertEquals("company1",resultDf.iloc[0]['ticker_symbol'])  
        self.assertEquals("company2",resultDf.iloc[1]['ticker_symbol'])  
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()