'''
Created on 10.03.2023

@author: vital
'''
import unittest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class TestVaderSentiment(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def test_sentiment_score(self):
        sentence = "The stock market is showing signs of recovery after the recent downturn."
        scores = self.analyzer.polarity_scores(sentence)
        self.assertAlmostEqual(scores['pos'], 0.0, places=3)
        self.assertAlmostEqual(scores['neg'], 0.0, places=3)
        self.assertAlmostEqual(scores['neu'], 1.0, places=3)
        self.assertAlmostEqual(scores['compound'], 0.0, places=3)
        