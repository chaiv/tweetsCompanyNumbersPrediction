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
        print(NearDuplicateDetector(df).getOriginalAndDuplicateRowsText())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()