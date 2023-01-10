'''
Created on 26.12.2022

@author: vital
'''
import unittest
import datetime
from tweetpreprocess.DateToTSP import DateTSPConverter

class DateTSPConverterTest(unittest.TestCase):


    def testWhenCorrectDateStrThenReturnTSP(self):
        self.assertEqual(1640991600, DateTSPConverter().dateStrToTSPInt("01/01/2022"))
        self.assertEqual(1640991600, DateTSPConverter("%d-%m-%y").dateStrToTSPInt("01-01-22"))
        self.assertEqual(1640991600, DateTSPConverter("%d.%m.%Y").dateStrToTSPInt("01.01.2022"))        
        pass
    
    def testWhenCorrectTSPIntThenReturnDate(self):
        self.assertEqual(datetime.datetime(2017, 1, 1, 1, 31,0),DateTSPConverter().tspIntToDate(1483230660))   
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()