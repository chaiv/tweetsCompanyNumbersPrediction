'''
Created on 24.02.2023

@author: vital
'''
import unittest
import torch
from unittest.mock import MagicMock
from classifier.transformer.Predictor import Predictor
from nlpvectors.TweetGroup import TweetGroup
from nlpvectors.AbstractEncoder import AbstractEncoder
from classifier.PredictionClassMappers import BINARY_0_1


class TweetGroupFakeWithTwoSentences:
    def getSeparatorIndexesInFeatureVector(self):
        return [2, 5]
 
    def getTokens(self):
        return [
             ["First", "tweet"],
             ["Second", "tweet"]
         ]
        
    def getTokenIndexes(self):
        return [
             [0,1],
             [3,4]
         ]
        
    def getSentenceIds(self): 
        return [0,1]
        
    def getFeatureVector(self):
        return [[0,0,0],[1,1,1],[2,2,2,],[0,0,0],[1,1,1],[2,2,2,]]

class TestEncoder(AbstractEncoder):
    
    def __init__(self):
        pass
    
    def getPADTokenID(self):
        return 0


class TestPredictor(unittest.TestCase):

    def setUp(self):
        self.model_mock = MagicMock()
        self.tokenizer_mock = MagicMock()
        self.encoder_mock = MagicMock()
        self.attributions_calculator_mock = MagicMock()
        self.prediction_class_mapper_mock = BINARY_0_1
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
        
        
    def test_calculateWordScores_of_tweetGroup(self):
        sentenceWrappers = [
            TweetGroupFakeWithTwoSentences(),
            TweetGroupFakeWithTwoSentences()
            ]
        self.encoder_mock.getPADTokenID = MagicMock(return_value=0)
        self.attributions_calculator_mock.attribute = MagicMock(return_value=
            torch.tensor(
                [
                    [1,1,0,2,2,0],
                    [1,1,0,2,2,0]
                ]
                )
            )
        result = self.predictor.calculateWordScoresOfTweetGroups(sentenceWrappers, observed_class=None, n_steps=None, internal_batch_size=None)
        self.assertEquals(2,len(result))
        self.assertEquals([0,1],result[0].getSentenceIds())
        self.assertEquals([[1, 1], [2, 2]],result[0].getAttributions())
        
    



        
        
            
    def test_calculate_attributions_of_tweetGroup(self):

        attributions_for_sentence_wrapper = torch.tensor([1,1,0,2,2,0])

        sentence_wrapper = TweetGroupFakeWithTwoSentences()

        wordScoresWrapper = self.predictor.calculateWordScoresOfTweetGroup(attributions_for_sentence_wrapper, sentence_wrapper)

        expected_attributions = [
            [1, 1],
            [2, 2]
        ]

        self.assertEqual(wordScoresWrapper.getAttributions(), expected_attributions)   
        
     
    def fake_model(self,x):
        x_list = x.tolist()
        if(len(x_list)==2 and x_list[0]==[1, 2, 0, 1, 2] and x_list[1]==[3, 1, 0, 1, 5]):
            return  torch.tensor([[0.7, 0.2], [0.3, 0.4]]) 
    
    def fake_model_for_chunk_prediction(self,x):
        x_list = x.tolist()
        if(len(x_list)==1 and x_list[0]==[1, 2, 0, 1, 2]):
            return  torch.tensor([[0.7, 0.2]])
        elif(len(x_list)==1 and x_list[0]==[3, 1, 0, 1, 5]):
            return  torch.tensor([[0.3, 0.4]])

         
        
    def test_predict_multiple_as_tweet_groups(self):
        tweetGroup1 = TweetGroup(sentences=["tweet 1", "second tweet"],sentenceIds=[123,456],totalTokenIndexes=[1,2,1,2],
                                 totalTokens=["tweet", "1", "second", "tweet"],
                                 totalFeatureVector=[1,2,0,1,2],separatorIndexesInFeatureVector=[2],label = 0)
        tweetGroup2 = TweetGroup(sentences=["next tweet", "tweet 4"],sentenceIds=[789,101112],totalTokenIndexes=[1,2,1,2],
                                 totalTokens=["next", "tweet", "tweet", "4"],
                                 totalFeatureVector=[3,1,0,1,5],separatorIndexesInFeatureVector=[2],label = 1)
        
        tweetGroups = [ tweetGroup1,tweetGroup2]

        predictor = Predictor(
            lambda x: self.fake_model(x), self.tokenizer_mock,TestEncoder(), self.prediction_class_mapper_mock,
            self.attributions_calculator_mock, self.device_to_use)
        
        result = predictor.predictMultipleAsTweetGroups(tweetGroups)
        self.assertEqual([tweetGroup1.getLabel(),tweetGroup2.getLabel()],result)
        
    def test_predict_multiple_as_tweet_groups_in_chunks(self):
        tweetGroup1 = TweetGroup(sentences=["tweet 1", "second tweet"],sentenceIds=[123,456],totalTokenIndexes=[1,2,1,2],
                                 totalTokens=["tweet", "1", "second", "tweet"],
                                 totalFeatureVector=[1,2,0,1,2],separatorIndexesInFeatureVector=[2],label = 0)
        tweetGroup2 = TweetGroup(sentences=["next tweet", "tweet 4"],sentenceIds=[789,101112],totalTokenIndexes=[1,2,1,2],
                                 totalTokens=["next", "tweet", "tweet", "4"],
                                 totalFeatureVector=[3,1,0,1,5],separatorIndexesInFeatureVector=[2],label = 1)
        
        tweetGroups = [ tweetGroup1,tweetGroup2]
        predictor = Predictor(
            lambda x: self.fake_model_for_chunk_prediction(x), self.tokenizer_mock,TestEncoder(), self.prediction_class_mapper_mock,
            self.attributions_calculator_mock, self.device_to_use)
        result = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1)
        self.assertEqual([tweetGroup1.getLabel(),tweetGroup2.getLabel()],result)

            

if __name__ == '__main__':
    unittest.main()