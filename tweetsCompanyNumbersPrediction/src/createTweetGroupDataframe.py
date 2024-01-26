'''
Created on 15.01.2024

@author: vital
'''
from nlpvectors.TweetGroupToDataframe import TweetGroupToDataframe
from tweetpreprocess.DataDirHelper import DataDirHelper
import pandas as pd
import numpy as np
from nlpvectors.DataframeSplitter import DataframeSplitter
import random
from classifier.ModelEvaluationHelper import createTweetGroupsAndTrueClasses
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer

tweetDf = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv")
tweetDf.fillna('', inplace=True) #nan values in body columns 
word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df.fillna('', inplace=True)
testSplitIndexes = np.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\test_idx_fold1.npy")
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        5,
        testSplitIndexes,
        tokenizer,
        textEncoder
        )
tweetGroupToDataframe = TweetGroupToDataframe()
tweetGroupDf = tweetGroupToDataframe.createTweetGroupDataframe(tweetGroups)
tweetGroupDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetGroups_at_5.csv")
