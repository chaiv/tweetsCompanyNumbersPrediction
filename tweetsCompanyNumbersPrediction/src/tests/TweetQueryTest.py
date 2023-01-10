'''
Created on 18.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from tweetpreprocess.DateToTSP import DateTSPConverter

class TweetQueryTest(unittest.TestCase):
    
    def testWhenQueryParamDateThenReturnThirdRow(self):
         
        dateToTSP = DateTSPConverter()
        
          
        tweetsDf = pd.DataFrame(
                  [
                  (1, dateToTSP.dateStrToTSPInt("01/01/2022"), dateToTSP.dateStrToTSPInt("01/03/2022")),
                  (2, dateToTSP.dateStrToTSPInt("01/02/2022"), dateToTSP.dateStrToTSPInt("01/05/2022")),
                  (3, dateToTSP.dateStrToTSPInt("01/03/2022"), dateToTSP.dateStrToTSPInt("01/06/2022"))
                  ],
                  columns=['tweet_id','from_date','to_date']
                  )
        resultDf = TweetDataframeQuery().query(tweetsDf, TweetQueryParams(fromDateStr='01/03/2022',toDateStr ='01/07/2022'))
        self.assertEquals(1,len(resultDf.index))  
        self.assertEquals(3,resultDf.iloc[0]['tweet_id'])  
    
    
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