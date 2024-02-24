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
from PredictionModelPath import AMAZON_20

predictionModelPath = AMAZON_20


word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
df = pd.read_csv(predictionModelPath.getDataframePath())
df.fillna('', inplace=True)
testSplitIndexes = np.load(predictionModelPath.getModelPath()+'\\test_idx_fold0.npy')
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        predictionModelPath.getTweetGroupSize(),
        testSplitIndexes,
        tokenizer,
        textEncoder
        )
tweetGroupToDataframe = TweetGroupToDataframe()
tweetGroupDf = tweetGroupToDataframe.createTweetGroupDataframe(tweetGroups)
tweetGroupDf.to_csv(predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+".csv")
