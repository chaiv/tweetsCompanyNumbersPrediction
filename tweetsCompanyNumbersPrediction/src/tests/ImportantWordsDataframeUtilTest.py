'''
Created on 14.11.2023

@author: vital
'''
import pandas as pd
import unittest
from featureinterpretation.InterpretationDataframeUtil import addUntokenizedWordColumnFromTweetDf


class ImportantWordsDataframeUtilTest(unittest.TestCase):


    def testUntokenizedWordColumnFromTweetDf(self):
        tweetDf = pd.DataFrame(
                  [
                    (123,"First tweets are Cool"),
                    (456,"Second tweet sucks")
                  ],
                  columns=["tweet_id","body"]
                  ) 
        importantWordsDf = pd.DataFrame(
                  [
                    (123,1,"tweet"),
                    (123,3,"cool"),
                    (456,2,"suck")
                  ],
                  columns=["tweet_id","token_index","token"]
                  ) 
        
        importantWordsDfWithOriginalTokenColumn = addUntokenizedWordColumnFromTweetDf(tweetDf,importantWordsDf)
        self.assertEquals(importantWordsDfWithOriginalTokenColumn["original_token"].iloc[0], "tweets")
        self.assertEquals(importantWordsDfWithOriginalTokenColumn["original_token"].iloc[1], "Cool")
        self.assertEquals(importantWordsDfWithOriginalTokenColumn["original_token"].iloc[2], "sucks")
        
        
        
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()