'''
Created on 26.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.DateToTSP import DateToTSP
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector

class PipelineTest(unittest.TestCase):


    def testWhenAllPipelineStepsThenCompletedDf(self):
        
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
        
        tweetsWithTSP = DateToTimestampDataframeTransformer(
            dateColumnNames=["post_date"],
            tspColumnNames= ["post_tsp"],
            dateToTSP=DateToTSP(dateFormat='%d/%m/%Y %H:%M:%S')
            ).addTimestampColumns(tweets)
        
        figures =  pd.DataFrame(
                  [
                  ("31/12/2024","01/01/2025",939880000.15),
                  ("10/03/2023","20/03/2023",1200000000.15),
                  ("28/02/2023","02/03/2023",963800000.15),
                  ("01/02/2022","01/02/2022",11000200.15)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        
        figuresWithTSP = DateToTimestampDataframeTransformer().addTimestampColumns(figures)
        
        tweetsWithNumbes = TweetNumbersConnector(fromTSPColumn = 'from_tsp',toTSPColumn ='to_tsp',valueColumn = 'value',postTSPColumn = 'post_tsp').getTweetsWithNumbers(tweetsWithTSP, figuresWithTSP)
        
        
        print(tweetsWithNumbes)
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()