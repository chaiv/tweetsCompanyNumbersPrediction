'''
Created on 05.08.2022

@author: vital
'''
import unittest
from tweetpreprocess.wordfiltering.HyperlinkFilter import HyperlinkFilter

class  HyperlinkFilterTest(unittest.TestCase):


    def testName(self):
        text = "Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur"
        self.assertEqual("Images showing the first Battery Swap Station  via @imgur", HyperlinkFilter().filter(text))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()