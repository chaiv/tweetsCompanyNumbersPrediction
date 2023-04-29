'''
Created on 24.02.2023

@author: vital
'''
import unittest
import torch
from unittest.mock import MagicMock
from classifier.transformer.Predictor import Predictor


class DummySentenceWrapper:
    def getSeparatorIndexesInFeatureVector(self):
        return [2, 5]
 
    def getTokens(self):
        return [
             ["First", "tweet"],
             ["Second", "tweet"]
         ]
        
    def getFeatureVector(self):
        return [[0,0,0],[1,1,1],[2,2,2,],[0,0,0],[1,1,1],[2,2,2,]]

class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.model_mock = MagicMock()
        self.tokenizer_mock = MagicMock()
        self.encoder_mock = MagicMock()
        self.attributions_calculator_mock = MagicMock()
        self.prediction_class_mapper_mock = MagicMock()
        self.device_to_use = "cuda:0"
        self.predictor = Predictor(
            self.model_mock, self.tokenizer_mock,self.encoder_mock, self.prediction_class_mapper_mock,
            self.attributions_calculator_mock, self.device_to_use)

    def test_calculateWordScores(self):
        sentences = ["The quick brown fox jumps over the lazy dog."]
        observed_class = 1
        n_steps = 500
        internal_batch_size = 10
        self.tokenizer_mock.tokenizeWithIndex = MagicMock(return_value=([0, 1, 2, 3, 4, 5, 6, 7, 8], ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog.']))
        self.encoder_mock.encodeTokens = MagicMock(return_value=[101, 1103, 1393, 3517, 4653, 1211, 1103, 12903, 4210, 119])
        self.tokenizer_mock.getPADTokenID = MagicMock(return_value=0)
        self.attributions_calculator_mock.attribute = MagicMock(return_value=[
            torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
            ])
        expected_scores = (
            ([0, 1, 2, 3, 4, 5, 6, 7, 8],),
            (['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog.'],),
            ([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        )
        result = self.predictor.calculateWordScores(sentences, observed_class, n_steps, internal_batch_size)
        self.assertEqual(expected_scores,result)
        
        
    def test_calculateWordScores_of_sentence_wrapper(self):
        sentenceWrappers = [
            DummySentenceWrapper(),
            DummySentenceWrapper()
            ]
        self.encoder_mock.getPADTokenID = MagicMock(return_value=0)
        self.attributions_calculator_mock.attribute = MagicMock(return_value=[
            torch.tensor([0.1,0.1,0.0,0.2,0.2,0.0])
            ])
        result = self.predictor.calculateWordScoresForWrappers(sentenceWrappers, observed_class=None, n_steps=None, internal_batch_size=None)
        self.assertIsNone(result)
        
        
            
    def test_calculate_attributions_of_sentence_wrapper(self):

        attributions_for_sentence_wrapper = [0.1,0.1,0.0,0.2,0.2,0.0]

        sentence_wrapper = DummySentenceWrapper()

        calculated_attributions = self.predictor.calculateAttributionsOfSentenceWrapper(attributions_for_sentence_wrapper, sentence_wrapper)

        expected_attributions = [
            [0.1, 0.1],
            [0.2, 0.2]
        ]

        self.assertEqual(calculated_attributions, expected_attributions)    
        

if __name__ == '__main__':
    unittest.main()