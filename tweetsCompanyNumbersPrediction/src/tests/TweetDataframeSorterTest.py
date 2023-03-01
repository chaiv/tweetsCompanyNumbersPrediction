'''
Created on 06.01.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.TweetDataframeSorter import TweetDataframeSorter


class ClassificationMetricsTest(unittest.TestCase):


    def testSortPostTSPAsc(self):
        df =  pd.DataFrame(
                  [
                  (3),
                  (1)
                  ],
                  columns=["post_tsp"]
                  )
        sortedDf = TweetDataframeSorter().sortByPostTSPAsc(df)
        self.assertEqual(1, sortedDf.iloc[0]["post_tsp"])
        self.assertEqual(3, sortedDf.iloc[1]["post_tsp"])
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'ClassificationMetricsTest.testSortPostTSPAsc']
    unittest.main()