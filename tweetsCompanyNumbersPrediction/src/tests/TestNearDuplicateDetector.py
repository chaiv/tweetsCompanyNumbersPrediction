'''
Created on 06.08.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.nearduplicates.NearDuplicateDetector import NearDuplicateDetector


class TestNearDuplicateDetector(unittest.TestCase):


    def testGetDuplicateIndexes(self):
        df = pd.DataFrame(
                  [
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at"),
                    ("Completely different tweet"),
                    ("have two apples and 2 oranges https://google.com")
                  ],
                  columns=["body"]
                  ) 
        nearDuplicates = NearDuplicateDetector(df).geDuplicateRowIndexes()
        self.assertEqual({1,4}, nearDuplicates)
        
    def testWrongDuplicates(self):
        df = pd.DataFrame(
                  [
                    ("$AMZN closed my puts for breakeven avg, volcrush from earlier in the week got me. scratched trade."),
                    ("Not with $AMZN 's much superior margins. Analysts need to upgrade on this. And on flufflky toy sales, as well."),
                  ],
                  columns=["body"]
                  ) 
        
        
        nearDuplicates = NearDuplicateDetector(df).geDuplicateRowIndexes()
        self.assertEqual(0, len(nearDuplicates))
            
    
        
    def testGetOriginalAndDuplicateIndexes(self):
        df = pd.DataFrame(
                  [
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("I have two apples and 2 oranges at"),
                    ("Completely different tweet"),
                    ("have two apples and 2 oranges https://google.com")
                  ],
                  columns=["body"]
                  ) 
        indexes = NearDuplicateDetector(df).getOriginalRowsWithDuplicateRowIndexesDefault()
        self.assertEqual(5, len(indexes))
        self.assertEqual([0,1,4], indexes[0])
        self.assertEqual([3], indexes[3])
        originalAndduplicateTexts = NearDuplicateDetector(df).getOriginalAndDuplicateRowsText()
        self.assertEqual(1, len(originalAndduplicateTexts))
        self.assertEqual(df["body"].iloc[0], originalAndduplicateTexts[0][0])
        self.assertEqual(df["body"].iloc[1], originalAndduplicateTexts[0][1])
        self.assertEqual(df["body"].iloc[4], originalAndduplicateTexts[0][2])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()