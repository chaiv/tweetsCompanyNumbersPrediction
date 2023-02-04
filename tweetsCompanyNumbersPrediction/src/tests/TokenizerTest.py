'''
Created on 03.02.2023

@author: vital
'''
import unittest
from classifier.transformer.nlp_utils import tokenize
from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec


class Test(unittest.TestCase):
    
    tokenizerTop2Vec = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAAPLFirst1000.json")

    def testNlpUtilTokenizer(self):
        text = "lx21 made $10,008  on $AAPL -Check it out!  Learn #howtotrade  $EXE $WATT $IMRS $CACH $GMO"
        self.assertIsNotNone(tokenize(text).tokens)
        self.assertIsNotNone(tokenize(text).ids)
        pass

    def testTop2VecTokenizer(self):
        text = "lx21 made $10,008  on $AAPL -Check it out!  Learn #howtotrade  $EXE $WATT $IMRS $CACH $GMO"
        self.assertEquals(2706,self.tokenizerTop2Vec.getVocabularyLength())
        self.assertEquals(2707,self.tokenizerTop2Vec.getPADTokenID())
        self.assertIsNotNone(self.tokenizerTop2Vec.encode(text))
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()