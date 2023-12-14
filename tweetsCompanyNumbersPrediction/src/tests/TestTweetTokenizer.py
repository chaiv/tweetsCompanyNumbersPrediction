'''
Created on 07.03.2023

@author: vital
'''
import unittest
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer

class TestTweetTokenizer(unittest.TestCase):
    
    def test_tokenize(self):
        word_filter = DefaultWordFilter()
        tokenizer = TweetTokenizer(word_filter)
        text = "This is a tweet with some stopwords and #hashtags and @mentions"
        expected_tokens = ["tweet", "stopword", "hashtag", "mention"]
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        
    def test_tokenizeWithIndex(self):
        word_filter = DefaultWordFilter()
        tokenizer = TweetTokenizer(word_filter)
        text = "This is a tweet with some stopwords and #hashtags and @mentions"
        expected_indexes = [3, 6, 8, 10]
        expected_tokens = ["tweet", "stopword", "hashtag", "mention"]
        indexes, tokens = tokenizer.tokenizeWithIndex(text)
        self.assertEqual(indexes, expected_indexes)
        self.assertEqual(tokens, expected_tokens)
        

    def test_tokenizeAndGetString(self):
        word_filter = DefaultWordFilter()
        tokenizer = TweetTokenizer(word_filter)
        text = "This is a tweet with some stopwords and #hashtags and @mentions"
        self.assertEqual("tweet stopword hashtag mention", tokenizer.tokenizeAndGetString(text))

if __name__ == '__main__':
    unittest.main()