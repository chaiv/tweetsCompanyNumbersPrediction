'''
Created on 11.04.2024

@author: vital
'''
from nlpvectors.TweetGroupToDataframe import TweetGroupToDataframe
from tweetpreprocess.DataDirHelper import DataDirHelper
import pandas as pd
import numpy as np
from nlpvectors.DataframeSplitter import DataframeSplitter
import random
from classifier.ModelEvaluationHelper import createTweetGroupsAndTrueClasses,\
    loadModel
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from PredictionModelPath import AMAZON_REVENUE_20, TESLA_CAR_SALES_5,\
    AMAZON_REVENUE_10
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from tweetpreprocess.LoadTweetDataframe import LoadTweetDataframe
from classifier.PredictionClassMappers import BINARY_0_1
from classifier.transformer.Predictor import Predictor
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
from featureinterpretation.TokenScoresSort import TokenScoresSort


predictionModelPath =  AMAZON_REVENUE_10
fold = 1
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
modelPath = predictionModelPath.getModelPath()+"\\tweetpredict_fold"+str(fold)+".ckpt"
model = loadModel(modelPath,word_vectors,evalMode=False)
df = LoadTweetDataframe(predictionModelPath).readDataframe()
testIdxPath = predictionModelPath.getModelPath()+'\\test_idx_fold'+str(fold)+'.npy'
testSplitIndexes = np.load(testIdxPath)
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        predictionModelPath.getTweetGroupSize(),
        testSplitIndexes,
        tokenizer,
        textEncoder
        )
tweetGroups = tweetGroups[:100]
predictor = Predictor(model,tokenizer ,textEncoder,BINARY_0_1,AttributionsCalculator(model,model.embedding))
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups,1000)
wordScoresWrappers = predictor.calculateWordScoresOfTweetGroupsInChunks(tweetGroups,observed_class=1,chunkSize=100,n_steps=500,internal_batch_size = 100)

tweetGroupColumn = "tweet_group"
labelColumn = "label"
tokensColumn = "tokens"
scoresColumn = "scores"

tweetGroupsLists = []
labelsLists = []
tokensLists = []
scoresLists = []

sorter = TokenScoresSort()


for wordScoreWrapper in wordScoresWrappers:
    labelsLists.append(wordScoreWrapper.getTweetGroup().getLabel())
    tweetGroupsLists.append(";".join(wordScoreWrapper.getTweetGroup().getSentences()))

    




print(wordScoresWrappers)
