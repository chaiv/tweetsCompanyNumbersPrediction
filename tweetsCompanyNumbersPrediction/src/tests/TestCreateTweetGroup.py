'''
Created on 23.04.2023

@author: vital
'''
import unittest
from unittest.mock import MagicMock
from nlpvectors.TweetGroup import createTweetGroup
from nlpvectors.VocabularyCreator import SEP_TOKEN
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.AbstractEncoder import AbstractEncoder

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

class TestCreateTweetGroup(unittest.TestCase):
    
    
    def test_zero_sentences(self):
        sentences = []
        sentence_ids = []
        label = None
        sentences_wrapper = createTweetGroup(FakeTokenizer(), FakeTextEncoder(), sentences, sentence_ids,label)
        self.assertEqual([],sentences_wrapper.totalTokenIndexes)
        self.assertEqual([],sentences_wrapper.totalTokens)
        self.assertEqual([],sentences_wrapper.sentenceIds)
        self.assertEqual ([], sentences_wrapper.separatorIndexesInFeatureVector)
        self.assertEqual([],sentences_wrapper.getFeatureVector())
        self.assertIsNone(sentences_wrapper.getLabel())
    
    def test_one_sentence(self):
        sentences = ['First tweet']
        sentence_ids = [0]
        label = 0
        sentences_wrapper = createTweetGroup(FakeTokenizer(), FakeTextEncoder(), sentences, sentence_ids,label)
        self.assertEqual([[0,0]],sentences_wrapper.totalTokenIndexes)
        self.assertEqual([['First', 'tweet']],sentences_wrapper.totalTokens)
        self.assertEqual([0],sentences_wrapper.sentenceIds)
        self.assertEqual ([], sentences_wrapper.separatorIndexesInFeatureVector)
        self.assertEqual([1, 1],sentences_wrapper.getFeatureVector())
        self.assertEqual(label,sentences_wrapper.getLabel())
        
    def test_two_sentences(self):
        sentences = ['First tweet','Second tweet']
        sentence_ids = [0,1]
        label = 0
        sentences_wrapper = createTweetGroup(FakeTokenizer(), FakeTextEncoder(), sentences, sentence_ids,label)
        self.assertEqual([[0,0],[0,0]],sentences_wrapper.totalTokenIndexes)
        self.assertEqual([['First', 'tweet'],['Second', 'tweet']],sentences_wrapper.totalTokens)
        self.assertEqual([0,1],sentences_wrapper.sentenceIds)
        self.assertEqual ([2], sentences_wrapper.separatorIndexesInFeatureVector)
        self.assertEqual([1, 1,2,1,1],sentences_wrapper.getFeatureVector())
        self.assertEqual(label,sentences_wrapper.getLabel())
