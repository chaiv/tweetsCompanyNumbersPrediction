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
                  (1.0),
                  (2.0),
                  (2.0),
                  (2.0)
                  ],
                  columns=["class"]
                  )
        value_counts = TweetDataframeExplore(df).getClassDistribution() #descending
        self.assertEqual([3,1],list(value_counts))
        pass
    
    def testWordCounts(self):
        df =  pd.DataFrame(
                  [
                  ("Hehe hello test"),
                  ("tweet data hehe"),
                  ("random word"),
                  (1.00),
                  ()
                  ],
                  columns=["body"]
                  )
        value_counts = TweetDataframeExplore(df).getMostFrequentWords(3) #descending
        self.assertEqual([2,1,1],list(value_counts))
        pass
    
    def testTweetsPerDay(self):
        df = pd.DataFrame(
                  [
                  (1483230660),
                  (1451607900),
                  (1420070457),
                  (1483230660),
                  (1420070457)
                  ],
                  columns=["post_date"]
                  )
    
        tweetsPerDay,_,_,_ = TweetDataframeExplore(df).getTweetsPerDayValues()
        self.assertEqual(732,len(tweetsPerDay))
        self.assertEqual(2,tweetsPerDay.iloc[0])
        
    def testNumberOfWordsValues(self):
        df = pd.DataFrame(
                  [
                  ("I like python"),
                  ("Pandas is really great"),
                  ("Matplotlib makes plots very easy")
                  ],
                  columns=["body"]
                  )
        word_counts, min_val,max_val,average = TweetDataframeExplore(df).getNumberOfWordsValues()
        self.assertEqual(3,word_counts.iloc[0])
        self.assertEqual(3,min_val)
        self.assertEqual(5,max_val)
        self.assertEqual(4,average)
        
    def testMostFrequentNamedEntities(self):
        df = pd.DataFrame(
                  [
                  ("I like Apple"),
                  ("Pandas is really great and Apple not"),
                  ("Matplotlib makes plots very easy")
                  ],
                  columns=["body"]
                  )
        frNE = TweetDataframeExplore(df).getMostFrequentWordsNamedEntities(3)
        self.assertEqual(3,len(frNE))
        self.assertEqual(('Apple', 2),frNE[0])
        self.assertEqual(('Pandas', 1),frNE[1])
        self.assertEqual(('Matplotlib', 1),frNE[2])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()