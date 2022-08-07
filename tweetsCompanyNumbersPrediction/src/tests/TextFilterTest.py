'''
Created on 07.08.2022

@author: vital
'''
import unittest
from tweetpreprocess.wordfiltering.StopWordsFilter import StopWordsFilter
from tweetpreprocess.wordfiltering.TextFilter import TextFilter

class TextFilterTest(unittest.TestCase):


    def testName(self):
        text = "The images showing the first Battery Swap Station via @imgur to them"
        self.assertEqual("The images showing Battery Swap Station @imgur", TextFilter([StopWordsFilter()]).filter(text))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()