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
        
    def test_tokenizer(self):
        word_filter = DefaultWordFilter()
        tokenizer = TweetTokenizer(word_filter)
        print(tokenizer.tokenize("lx21 made $10,008  on $AAPL -Check it out!  Learn #howtotrade  $EXE $WATT $IMRS $CACH $GMO"))

if __name__ == '__main__':
    unittest.main()