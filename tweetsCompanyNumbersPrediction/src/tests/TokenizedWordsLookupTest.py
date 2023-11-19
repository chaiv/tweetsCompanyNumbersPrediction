'''
Created on 19.11.2023

@author: vital
'''
import unittest
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup


class TokenizedWordsLookupTest(unittest.TestCase):


    def testLookUp(self):
        sentence = "This is a tweet with some stopwords and #hashtags and @mentions"
        expected_tokens = ["tweet", "stopword", "hashtag", "mention"]
        lookupDict = TokenizedWordsLookup().createLookUpDictionary([sentence])
        print(lookupDict)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()