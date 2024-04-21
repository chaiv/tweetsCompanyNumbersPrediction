'''
Created on 19.11.2023

@author: vital
'''
import unittest
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup,\
    createLookUpDictionary


class TokenizedWordsLookupTest(unittest.TestCase):


    def testLookUp(self):
        sentence = "This is a tweet with some stopwords and #hashtags and @mentions"
        lookupDict = createLookUpDictionary([sentence])
        self.assertEquals({'tweet': 'tweet', 'stopword': 'stopwords', 'hashtag': '#hashtags', 'mention': '@mentions'},lookupDict)
        wordsLookup = TokenizedWordsLookup(tokenizerLookupDict = lookupDict)
        self.assertTrue(wordsLookup.hasOriginalWord('mention'))
        self.assertEquals('@mentions',wordsLookup.getOriginalWord('mention'))
        
    def testNone(self):
        wordsLookup = TokenizedWordsLookup(tokenizerLookupDict = {})
        self.assertEquals([None],wordsLookup.getOriginalWords(['mention']))
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()