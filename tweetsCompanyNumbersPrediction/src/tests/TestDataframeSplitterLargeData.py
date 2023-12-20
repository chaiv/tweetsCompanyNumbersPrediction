'''
Created on 19.10.2023

@author: vital
'''
import unittest
from tweetpreprocess.DataDirHelper import DataDirHelper

import pandas as pd
from nlpvectors.DataframeSplitter import DataframeSplitter

class TopicExtractorBertTest(unittest.TestCase):


    def testDfWithGroupedTweets(self):
        df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv") 
        df.fillna('', inplace=True) #nan values in body columns 
        resultDf =  DataframeSplitter().getDfWithGroupedTweets(df,5)
        print(resultDf)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TopicExtractorBertTest.testName']
    unittest.main()