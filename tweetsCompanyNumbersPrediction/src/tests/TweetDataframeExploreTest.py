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
        self.assertEqual(732,len(tweetsPerDay)) #732 for every day
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
        
    def testFrequentNamedEntities(self):
        df = pd.DataFrame(
                  [
                  ("I like Apple"),
                  ("Pandas is really great and Apple not"),
                  ("Matplotlib makes plots very easy")
                  ],
                  columns=["body"]
                  )
        mfrNE = TweetDataframeExplore(df).getMostFrequentWordsNamedEntities(3)
        lfrNE = TweetDataframeExplore(df).getLeastFrequentWordNamedEntities(3)
        self.assertEqual(3,len(mfrNE))
        self.assertEqual(('Apple', 2),mfrNE[0])
        self.assertEqual(('Pandas', 1),mfrNE[1])
        self.assertEqual(('Matplotlib', 1),mfrNE[2])
        self.assertEqual(('Matplotlib', 1),lfrNE[0])
        
    def testCardinalNumbersPerTweetValues(self):
        df = pd.DataFrame(
                  [
                  (1483230660,"I have two apples and 2 oranges"),
                  (1483230660,"9.87 ways to go")
                  ],
                  columns=["post_date","body"]
                  )
        counts, min_val,max_val,average = TweetDataframeExplore(df).getCardinalNumbersPerTweetValues()
        self.assertEqual(2,len(counts))
        self.assertEqual(2,counts[0])
        self.assertEqual(1,counts[1])
        self.assertEqual(1,min_val)
        self.assertEqual(2,max_val)
        self.assertEqual(1.5,average)
        
    def testURLsPerTweetValues(self):
        df = pd.DataFrame(
                  [
                    ("I have two apples and 2 oranges at https://google.com"),
                    ("9.87 ways to go")
                  ],
                  columns=["body"]
                  )   
        counts, min_val,max_val,average = TweetDataframeExplore(df).getURLPerTweetValues()
        self.assertEqual(1,counts[0])
        self.assertEqual(0,counts[1])
        self.assertEqual(1,max_val)
        self.assertEqual(0,min_val)
        self.assertEqual(0.5,average)
        
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
        originalAndDuplicateTexts = TweetDataframeExplore(df).getOriginalAndNearDuplicateRowsText()
        self.assertEqual(1,len(originalAndDuplicateTexts))
        self.assertEqual(df["body"].iloc[0],originalAndDuplicateTexts[0][0])
        self.assertEqual(df["body"].iloc[4],originalAndDuplicateTexts[0][1]) 
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()