'''
Created on 18.08.2023

@author: vital
'''
import unittest
import pandas as pd
from sentiment.TweetSentimentAnalysis import TweetSentimentAnalysis,\
    POSITIVE_LABEL, NEUTRAL_LABEL, NEGATIVE_LABEL

class TweetSentimentAnalysisTest(unittest.TestCase):


    def testSentiment(self):
        df = pd.DataFrame(
                  [
                    ("Review: iOgrapher iPhone and iPad camera platform  #AppleInsider $AAPL"),
                    ("Apple faces lawsuit over massive storage space required by iOS 8 $AAPL "),
                    ("I am proud to report that emerging markets wireless operators are increasing their support for the iPhone 6."),
                  ],
                  columns=["body"]
                  ) 
        dfWithSentiment = TweetSentimentAnalysis(df).getDfWithSentiment()
        self.assertEqual(NEUTRAL_LABEL, dfWithSentiment["sentiment_label"].iloc[0])
        self.assertEqual(NEGATIVE_LABEL, dfWithSentiment["sentiment_label"].iloc[1])
        self.assertEqual(POSITIVE_LABEL, dfWithSentiment["sentiment_label"].iloc[2])
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()