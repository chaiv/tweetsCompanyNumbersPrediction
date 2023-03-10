'''
Created on 10.03.2023

@author: vital
'''

import unittest
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.VocabularyCreator import VocabularyCreator


class TestVocabularyCreator(unittest.TestCase):
    def setUp(self):
        
        class Tokenizer(AbstractTokenizer):
            def tokenize(self,sentence):
                return sentence.split(" ")
        self.vocabulary_creator = VocabularyCreator(Tokenizer())

    def test_createVocabularyDict(self):
        sentences = ["This is a test sentence", "This is another test sentence"]
        expected_vocab = {'This': 2, 'is': 3, 'test': 4, 'sentence': 5, 'a': 6, 'another': 7, '<PAD>': 0, '<UNK>': 1}
        vocab = self.vocabulary_creator.createVocabularyDict(sentences)
        self.assertEqual(vocab, expected_vocab)


if __name__ == '__main__':
    unittest.main()
