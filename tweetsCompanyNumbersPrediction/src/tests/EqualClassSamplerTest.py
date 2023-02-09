'''
Created on 09.02.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.EqualClassSampler import EqualClassSampler

class EqualClassSamplerTest(unittest.TestCase):


    def testSampler(self):
        df =  pd.DataFrame(
                  [
                  (1.0),
                  (1.0),
                  (2.0),
                  (2.0),
                  (2.0)
                  ],
                  columns=["class"]
                  )
        
        resultDf = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
        self.assertEqual(4, len(resultDf.index))
        self.assertEqual(2, len(resultDf[resultDf["class"]==2.0].index))
        self.assertEqual(2, len(resultDf[resultDf["class"]==1.0].index))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()