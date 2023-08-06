'''
Created on 06.08.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.nearduplicates.NearDuplicateDetector import NearDuplicateDetector

class TestNearDuplicateDetector(unittest.TestCase):


    def testDetector(self):
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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()