'''
Created on 07.01.2023

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.FiguresIncreaseDecreaseClassCalculator import FiguresIncreaseDecreaseClassCalculator

class FiguresIncreaseDecreaseClassCalculatorTest(unittest.TestCase):


    def testIncreaseDecreaseClasses(self):
        figures =  pd.DataFrame(
                  [
                  (1.12),
                  (0.857),
                  (1),
                  (0.0)
                  ],
                  columns=['percentChange']
                  )
        resultDf = FiguresIncreaseDecreaseClassCalculator().getFiguresWithClasses(figures);
        self.assertEqual(1,resultDf.iloc[0]["class"])
        self.assertEqual(0,resultDf.iloc[1]["class"])
        self.assertEqual(0,resultDf.iloc[2]["class"])
        self.assertEqual(0,resultDf.iloc[3]["class"])
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()