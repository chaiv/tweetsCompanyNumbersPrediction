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
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.BinaryClassificationMetricsPlots import BinaryClassificationMetricsPlots
from classifier.ModelEvaluationHelper import loadModel,\
    createTweetGroupsAndTrueClasses
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler

from PredictionModelPath import AMAZON_REVENUE_5, MICROSOFT_EPS_5,\
    MICROSOFT_GROSS_PROFIT_20, MICROSOFT_XBOX_USERS_20, MICROSOFT_XBOX_USERS_10,\
    GOOGLE_SE_MARKET_SHARE_10
from PredictionModelPath import AMAZON_REVENUE_10
from PredictionModelPath import AMAZON_REVENUE_20
from PredictionModelPath import APPLE_IPHONE_SALES_5
from PredictionModelPath import APPLE_IPHONE_SALES_10
from PredictionModelPath import APPLE__IPHONE_SALES_20
from PredictionModelPath import TESLA_CAR_SALES_5
from PredictionModelPath import TESLA_CAR_SALES_10
from PredictionModelPath import TESLA_CAR_SALES_20
from tweetpreprocess.LoadTweetDataframe import LoadTweetDataframe

predictionModelPath =  GOOGLE_SE_MARKET_SHARE_10
fold = 1
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
modelPath = predictionModelPath.getModelPath()+"\\tweetpredict_fold"+str(fold)+".ckpt"
model = loadModel(modelPath,word_vectors)
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
predictor = Predictor(model,tokenizer,textEncoder ,BINARY_0_1 ,None)
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
print("true_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
print("prediction_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(prediction_classes).items()))
metrics = BinaryClassificationMetrics() 
plots = BinaryClassificationMetricsPlots(metrics)
print(metrics.classification_report(trueClasses, prediction_classes))
print("MCC "+str(metrics.calculate_mcc(trueClasses, prediction_classes)))
BinaryClassificationMetricsPlots(metrics).createRocAUCAndPrAucPlots(trueClasses, prediction_classes)

