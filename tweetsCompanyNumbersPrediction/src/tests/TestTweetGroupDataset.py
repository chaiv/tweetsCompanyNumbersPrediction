'''
Created on 25.04.2023

@author: vital
'''
import unittest
import pandas as pd
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder
from classifier.TweetGroupDataset import TweetGroupDataset
import torch
from nlpvectors.DataframeSplitter import DataframeSplitter

class FakeTokenizer(AbstractTokenizer):
    def __init__(self):
        pass
    
    def tokenizeWithIndex(self, text):
        indexes = []
        tokens = []
        for token in text.split(' '):
            indexes.append(0)
            tokens.append(token)
        return indexes,tokens

class FakeTextEncoder(AbstractEncoder):
    def __init__(self):
        pass
    
    def encodeTokens(self, tokens):
        encoded = []
        for token in tokens:
            encoded.append(1)
        return encoded

    def getSEPTokenID(self):
        return 2

class TestTweetGroupDataset(unittest.TestCase):
      
    def setUp(self):
        data = {
            'body': ['tweet 1', 'tweet 2', 'tweet 3', 'tweet 4','tweet 5'],
            'class': [0, 0, 1, 1,0]
        }
        df = pd.DataFrame(data)
        tokenizer = FakeTokenizer()
        textEncoder = FakeTextEncoder()
        samples = DataframeSplitter().getDfSplitIndexes(df, 2, 'class')
        self.dataset = TweetGroupDataset(df, samples, tokenizer, textEncoder)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_getitem_multiple_tweets(self):
        x, y = self.dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertTrue(x.dtype == torch.long)
        expected_x = torch.tensor([1,1,2,1,1])
        expected_y = 0
        self.assertTrue(torch.equal(x, expected_x))
        self.assertEqual(y, expected_y)
        
    def test_getitem_remaining_tweet(self):
        x, y = self.dataset[1] # 1 and not 2, because it is sorted by class labels and the last remaining sentence with class 0 is put as own sample after the first two with class 0. 
        self.assertIsInstance(x, torch.Tensor)
        self.assertTrue(x.dtype == torch.long)
        expected_x = torch.tensor([1,1])
        expected_y = 0
        self.assertTrue(torch.equal(x, expected_x))
        self.assertEqual(y, expected_y)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()