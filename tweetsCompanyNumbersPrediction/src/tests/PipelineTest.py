'''
Created on 26.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.DateToTSP import DateTSPConverter
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer
from tweetnumbersconnector.tweetnumbersconnector import TweetNumbersConnector
from nlpvectors.tfidfVectorizer import TFIDFVectorizer
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter
from tweetpreprocess.TweetTextFilterTransformer import TweetTextFilterTransformer
from tweetpreprocess.FiguresPercentChangeCalculator import FiguresPercentChangeCalculator
from tweetpreprocess.FiguresIncreaseDecreaseClassCalculator import FiguresIncreaseDecreaseClassCalculator

class PipelineTest(unittest.TestCase):


    def testPipelineSteps(self):
        
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
                  ("01/01/2021 00:00:00","31/12/2022 23:59:59",10),#These dates are not in tweets, needed to calculate initial percent change
                  ("01/01/2022 00:00:00","01/01/2022 23:59:59",11),
                  ("28/02/2023 00:00:00","02/03/2023 23:59:59",8.15),
                  ("10/03/2023 00:00:00","20/03/2023 23:59:59",12.15),
                  ("31/12/2024 00:00:00","01/01/2025 23:59:59",9.15)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        
        tweetsWithTSP = DateToTimestampDataframeTransformer(
            dateColumnNames=["post_date"],
            tspColumnNames= ["post_tsp"],
            dateToTSP=DateTSPConverter(dateFormat=dateFormat)
            ).addTimestampColumns(tweets)
                
        figuresWithTSP = DateToTimestampDataframeTransformer(dateToTSP=DateTSPConverter(dateFormat=dateFormat)).addTimestampColumns(figures)
        
        figuresWithClasses =  FiguresIncreaseDecreaseClassCalculator().getFiguresWithClasses(FiguresPercentChangeCalculator ().getFiguresWithClasses(figuresWithTSP))
        
        tweetsWithNumbers = TweetNumbersConnector(
            valueColumn="class").getTweetsWithNumbers(tweetsWithTSP,  figuresWithClasses)
        
        textfiltetedTweetsWithNumbers  = TweetTextFilterTransformer(TextFilter([HyperlinkFilter()])).filterTextColumns(tweetsWithNumbers)  

        tfidfVectorizer = TFIDFVectorizer(textfiltetedTweetsWithNumbers )
        
        tweetsWithTFIDF = tfidfVectorizer.getTweetsWithTFIDFVectors()
                 
        self.assertEqual(1, tweetsWithTFIDF .iloc[0]["class"])    
        self.assertEqual(1, tweetsWithTFIDF .iloc[1]["class"]) 
        self.assertEqual(0, tweetsWithTFIDF .iloc[2]["class"])     
        self.assertEqual(1, tweetsWithTFIDF .iloc[3]["class"])        
        self.assertEqual(0, tweetsWithTFIDF .iloc[4]["class"])  
        
        self.assertFalse("http://imgur.com/dummylink" in tweetsWithTFIDF .iloc[0]['body'] )
        
        self.assertEqual( 0.34706676322953556, tweetsWithTFIDF.iloc[3]['tfidf'][0])
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()