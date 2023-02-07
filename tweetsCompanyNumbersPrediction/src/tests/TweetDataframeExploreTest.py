'''
Created on 07.02.2023

@author: vital
'''
import unittest
import pandas as pd
from exploredata.TweetDataframeExplore import TweetDataframeExplore


class TweetDataframeExploreTest(unittest.TestCase):


    def testClassCounts(self):
        df =  pd.DataFrame(
                  [
                  (1),
                  (2),
                  (2)
                  ],
                  columns=["class"]
                  )
        value_counts = TweetDataframeExplore(df).getClassDistribution() #descending
        self.assertEqual([2,1],list(value_counts))
        pass
    
    def testWordCounts(self):
        df =  pd.DataFrame(
                  [
                  ("Hehe hello test"),
                  ("tweet data hehe"),
                  ("random word")
                  ],
                  columns=["body"]
                  )
        value_counts = TweetDataframeExplore(df).getMostFrequentWords(3) #descending
        self.assertEqual([2,1,1],list(value_counts))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()