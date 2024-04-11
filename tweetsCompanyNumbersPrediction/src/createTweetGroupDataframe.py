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
from PredictionModelPath import AMAZON_REVENUE_20, TESLA_CAR_SALES_5,\
    AMAZON_REVENUE_10
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler


predictionModelPath = AMAZON_REVENUE_10


word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
df = pd.read_csv(predictionModelPath.getDataframePath())
df.fillna('', inplace=True)
df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
testSplitIndexes = np.load(predictionModelPath.getModelPath()+'\\test_idx_fold0.npy')
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        predictionModelPath.getTweetGroupSize(),
        testSplitIndexes,
        tokenizer,
        textEncoder
        )
print("true_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
tweetGroupToDataframe = TweetGroupToDataframe()
tweetGroupDf = tweetGroupToDataframe.createTweetGroupDataframe(tweetGroups)
tweetGroupDf.to_csv(predictionModelPath.getModelPath()+"\\tweetGroups_at_"+str(predictionModelPath.getTweetGroupSize())+".csv")
