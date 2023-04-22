'''
Created on 10.03.2023

@author: vital
'''

import unittest
from nlpvectors.AbstractTokenizer import AbstractTokenizer
from nlpvectors.VocabularyCreator import VocabularyCreator, UNK_TOKEN, PAD_TOKEN,\
    SEP_TOKEN


class TestVocabularyCreator(unittest.TestCase):
    def setUp(self):
        
        class Tokenizer(AbstractTokenizer):
            def tokenize(self,sentence):
                return sentence.split(" ")
        self.vocabulary_creator = VocabularyCreator(Tokenizer())

    def test_createVocabularyDict(self):
        sentences = ["This is a test sentence", "This is another test sentence"]
        expected_vocab = {'This': 0, 'is': 1, 'test': 2, 'sentence':3, 'a': 4, 'another': 5, PAD_TOKEN: 6, UNK_TOKEN: 7,SEP_TOKEN : 8}
        vocab = self.vocabulary_creator.createVocabularyDict(sentences)
        self.assertEqual(vocab, expected_vocab)


if __name__ == '__main__':
    unittest.main()
