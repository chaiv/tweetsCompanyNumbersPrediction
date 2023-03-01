'''
Created on 03.02.2023

@author: vital
'''
import unittest
from classifier.transformer.nlp_utils import tokenize
from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tagging.PosDepTagger import PosDepTagger


class ClassificationMetricsTest(unittest.TestCase):
    
    tokenizerTop2Vec = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAAPLFirst1000.json")


    def testTokenizedWithIndex(self):
        text = "Tesla is cool"
        result = self.tokenizerTop2Vec.tokenizeWithIndex(text)
        self.assertEqual(3,
            len(result[0]))
        self.assertEqual(3,
            len(result[1]))


    def testTokenizedWithTags(self):
        text = "Tesla is cool"
        result = self.tokenizerTop2Vec.tokenizeWithTagging(text, PosDepTagger())
        self.assertEqual(3,
            len(result))
        self.assertEqual('nsubj', result[0][3])
      

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

    def testGetWords(self):
        tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
        self.assertEquals([626, 36850],tokenizer.encode("Follow @StockMoney62"))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'ClassificationMetricsTest.testName']
    unittest.main()