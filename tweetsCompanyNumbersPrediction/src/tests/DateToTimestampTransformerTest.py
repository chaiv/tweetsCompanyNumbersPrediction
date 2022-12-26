'''
Created on 25.12.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.DateToTimestampTransformer import DateToTimestampDataframeTransformer


class DateToTimestampTransformerTest(unittest.TestCase):


        def testFiguresDiscretizer(self):
            testDf =  pd.DataFrame(
                  [
                  ("01/10/2014", "31/12/2014"),           
                  ("01/01/2015", "31/03/2015")
                  ],
                  columns=["from_date","to_date"]
                  )
            dfWithTSP = DateToTimestampDataframeTransformer().addTimestampColumns(testDf)
            self.assertEqual(2,len(dfWithTSP.index))
            self.assertEquals(1412114400,dfWithTSP.iloc[0]['from_tsp'])  
            self.assertEquals(1419980400,dfWithTSP.iloc[0]['to_tsp'])  


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()