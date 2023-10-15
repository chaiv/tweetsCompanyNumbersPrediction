'''
Created on 15.10.2023

@author: vital
'''
import unittest
import pandas as pd
from tests.TestTweetGroupDataset import FakeTokenizer, FakeTextEncoder
from classifier.TweetGroupDataset import TweetGroupDataset
from tweetpreprocess.DataDirHelper import DataDirHelper
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer


class TestTweetGroupDatasetLargeData(unittest.TestCase):
      
    def test_access_dataset(self):
        df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv") 
        word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
        textEncoder = WordVectorsIDEncoder(word_vectors)
        tokenizer = TweetTokenizer(DefaultWordFilter())
        n_tweets_as_sample = 5    
        dataset = TweetGroupDataset(df, n_tweets_as_sample, tokenizer, textEncoder)
        for i in range(dataset.__len__()):
            x,y= dataset.__getitem__(i)
            print(len(x))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()