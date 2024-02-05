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
                  (1,1.0),
                  (2,1.0),
                  (3,2.0),
                  (4,2.0),
                  (5,2.0)
                  ],
                  columns=["id","class"]
                  )
        
        resultDf = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
        self.assertEqual(4, len(resultDf.index))
        self.assertEqual(2, len(resultDf[resultDf["class"]==2.0].index))
        self.assertEqual(2, len(resultDf[resultDf["class"]==1.0].index))
        self.assertEqual(3,resultDf["id"].iloc[0])
        self.assertEqual(4,resultDf["id"].iloc[1])
        self.assertEqual(1,resultDf["id"].iloc[2])
        self.assertEqual(2,resultDf["id"].iloc[3])
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()