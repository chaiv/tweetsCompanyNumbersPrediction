'''
Created on 07.08.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.nearduplicates.DuplicateDetector import DuplicateDetector

class TestDuplicateDetector(unittest.TestCase):


    def testGetDuplicatesDf(self):
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
        duplicates = DuplicateDetector(df).getDuplicatesDataframe()
        self.assertEqual(1, len(duplicates))
        
    def testGetDfWithoutDuplicates(self):
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
        dfWithoutDuplicates = DuplicateDetector(df).getDataframeWithoutDuplicates()
        self.assertEqual(4, len(dfWithoutDuplicates))
    



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()