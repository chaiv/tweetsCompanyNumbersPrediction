'''
Created on 19.02.2023

@author: vital
'''
import unittest
from exploredata.POSTagging import PartOfSpeechTagging
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer



class TestPOSTagging(unittest.TestCase):
    
    def setUp(self):
        self.tagger = PartOfSpeechTagging(TweetTokenizer(DefaultWordFilter()))
        
    def test_tagging(self):
        sentence = "$AMZN - Amazon to Produce Original Movies for Theaters, Prime Instant Video "
        tags = self.tagger.getPOSTags(sentence)
        print(tags)