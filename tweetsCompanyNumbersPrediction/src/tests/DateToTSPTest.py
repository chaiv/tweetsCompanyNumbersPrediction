'''
Created on 26.12.2022

@author: vital
'''
import unittest
from tweetpreprocess.DateToTSP import DateToTSP

class DateToTSPTest(unittest.TestCase):


    def testWhenCorrectDateStrThenReturnTSP(self):
        self.assertEqual(1640991600, DateToTSP().dateStrToTSPInt("01/01/2022"))
        self.assertEqual(1640991600, DateToTSP("%d-%m-%y").dateStrToTSPInt("01-01-22"))
        self.assertEqual(1640991600, DateToTSP("%d.%m.%Y").dateStrToTSPInt("01.01.2022"))        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()