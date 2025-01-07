'''
Created on 09.03.2023

@author: vital
'''
import pandas as pd
import numpy as np
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from classifier.ClassificationMetrics import ClassificationMetrics
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.ModelEvaluationHelper import loadModel,\
    createTweetGroupsAndTrueClasses
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler

from tweetpreprocess.LoadTweetDataframe import LoadTweetDataframe
from classifier.PredictionClassMapper import PredictionClassMapper
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_MULTI_CLASS

predictionModelPath = AMAZON_REVENUE_10_LSTM_MULTI_CLASS
predictionClassMapper = AMAZON_REVENUE_10_LSTM_MULTI_CLASS.getPredictionClassMapper()
fold = 1
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
modelPath = predictionModelPath.getModelPath()+"\\tweetpredict_fold"+str(fold)+".ckpt"
model = loadModel(modelPath,word_vectors,num_classes=predictionClassMapper.get_number_of_classes())
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
predictor = Predictor(model,tokenizer,textEncoder ,predictionClassMapper ,None)
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
print("true_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
print("prediction_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(prediction_classes).items()))
metrics = ClassificationMetrics() 
print(metrics.classification_report(trueClasses, prediction_classes))
print("MCC "+str(metrics.calculate_mcc(trueClasses, prediction_classes)))
#BinaryClassificationMetricsPlots(metrics).createRocAUCAndPrAucPlots(trueClasses, prediction_classes)

