'''
Created on 08.05.2024

@author: vital
'''
import unittest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
words = "Poor, sour, fake, worst".split(",")
for word in words: 
    scores = analyzer.polarity_scores(word)
    print(word, "pos",scores['pos'],"neq",scores['neg'],"neu",scores['neu'], "comp",scores['compound'])




