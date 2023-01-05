'''
Created on 26.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.DateToTSP import DateToTSP
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector
from tweetpreprocess.TweetDataframeQuery import TweetDataframeQuery
from tweetpreprocess.TweetQueryParams import TweetQueryParams
from nlpvectors.tfidfVectorizer import TFIDFVectorizer

class PipelineTest(unittest.TestCase):


    def testWhenAllPipelineStepsThenCompletedDf(self):
        
        dateFormat='%d/%m/%Y %H:%M:%S'
        
        tweets = pd.DataFrame(
                  [
                  ("1","01/01/2022 12:11:11","Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur","company1"),
                  ("2","01/01/2022 12:12:12","company next charger will automatically connect to your car http://flip.it/dummylink","company2"),
                  ("3","01/03/2023 00:01:00","Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink","company1"),
                  ("4","20/03/2023 12:55:56","Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink","company2"),
                  ("5","31/12/2024 13:01:45","Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink","company1")
                  ],
                  columns=["tweet_id","post_date","body","ticker_symbol"]
                  )
        
        figures =  pd.DataFrame(
                  [
                  ("31/12/2024 00:00:00","01/01/2025 23:59:59",9.15),
                  ("10/03/2023 00:00:00","20/03/2023 23:59:59",12.15),
                  ("28/02/2023 00:00:00","02/03/2023 23:59:59",8.15),
                  ("01/01/2022 00:00:00","01/01/2022 23:59:59",11)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        
        tweetsWithTSP = DateToTimestampDataframeTransformer(
            dateColumnNames=["post_date"],
            tspColumnNames= ["post_tsp"],
            dateToTSP=DateToTSP(dateFormat=dateFormat)
            ).addTimestampColumns(tweets)
                
        figuresWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateToTSP(dateFormat=dateFormat)).addTimestampColumns(figures)
        
        tweetsWithNumbers = TweetNumbersConnector(
            postDateColumn ="post_date").getTweetsWithNumbers(tweetsWithTSP, figuresWithTSP)
            
        self.assertEqual(11, tweetsWithNumbers.iloc[0]["value"])    
        self.assertEqual(11, tweetsWithNumbers.iloc[1]["value"]) 
        self.assertEqual(8.15, tweetsWithNumbers.iloc[2]["value"])     
        self.assertEqual(12.15, tweetsWithNumbers.iloc[3]["value"])        
        self.assertEqual(9.15, tweetsWithNumbers.iloc[4]["value"])  

        tweetsFilteredByIds = TweetDataframeQuery().query( tweetsWithNumbers, TweetQueryParams(tweetIds=["2","4","5"]))
        
        self.assertEqual(3, len(tweetsFilteredByIds))  
        
        tfidfVectorizer = TFIDFVectorizer(tweetsFilteredByIds)
        
        tweetsWithTFIDF = tfidfVectorizer.getTweetsWithTFIDFVectors()
        
        self.assertEqual( 0.28456870659163674, tweetsWithTFIDF.iloc[0]['tfidf'][1])
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()