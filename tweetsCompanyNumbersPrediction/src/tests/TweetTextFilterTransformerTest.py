'''
Created on 26.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter
from tweetpreprocess.TweetTextFilterTransformer import TweetTextFilterTransformer

class TweetTextFilterTransformerTest(unittest.TestCase):


    def testWhenDfWithHyperlinksRemoveThem(self):
        tweetsDf = pd.DataFrame(
                  [
                  ("Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur"),
                  ("company next charger will automatically connect to your car http://flip.it/dummylink"),
                  ("Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink"),
                  ("Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink"),
                  ("Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink")
                  ],
                  columns=["body"]
                  )
        
        resultDf = TweetTextFilterTransformer(TextFilter([HyperlinkFilter()])).filterTextColumns(tweetsDf)  
        self.assertEqual(5,len(resultDf.index))  
        self.assertEqual("Images showing the first Battery Swap Station  via @imgur", resultDf.iloc[0]["body"] )
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()