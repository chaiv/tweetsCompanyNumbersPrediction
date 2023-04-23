'''
Created on 23.04.2023

@author: vital
'''
import unittest
from unittest.mock import MagicMock
from nlpvectors.SentenceWrapper import createSentencesWrapper
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

class TestCreateSentencesWrapper(unittest.TestCase):
    
    def test_one_sentence(self):
        sentences = ['First tweet']
        sentence_ids = [0]
        sentences_wrapper = createSentencesWrapper(FakeTokenizer(), FakeTextEncoder(), sentences, sentence_ids)

        self.assertEqual([[0,0]],sentences_wrapper.totalTokenIndexes)
        self.assertEqual([['First', 'tweet']],sentences_wrapper.totalTokens)
        self.assertEqual([[1, 1, 2]],sentences_wrapper.totalEncodedTokens)
        self.assertEqual([0],sentences_wrapper.sentenceIds)
        self.assertEqual([1, 1, 2],sentences_wrapper.getFeatureVector())
