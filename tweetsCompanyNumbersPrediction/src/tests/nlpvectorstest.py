'''
Created on 30.01.2022

@author: vital
'''
import unittest
import pandas as pd
from nlpvectors.tfidfVectorizer import TFIDFVectorizer

class NLPVectorsTest(unittest.TestCase):


    def testTFIDF(self):
        tweets = pd.DataFrame(
                  [
                  ("Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur"),
                  ("company next charger will automatically connect to your car http://flip.it/dummylink"),
                  ("Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink"),
                  ("Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink"),
                  ("Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink")
                  ],
                  columns=["body"]
                  )
        tfidfVectorizer = TFIDFVectorizer(tweets,tfidfvectorsColumnName='tfidfvectors')
        self.assertEqual( 54, tfidfVectorizer.getFeatureNames().__len__())
        self.assertEqual( 'accelerating', tfidfVectorizer.getFeatureNames()[0])
        self.assertEqual( 0.0, tfidfVectorizer.getTweetsWithTFIDFVectors().iloc[0]['tfidfvectors'][0])
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()