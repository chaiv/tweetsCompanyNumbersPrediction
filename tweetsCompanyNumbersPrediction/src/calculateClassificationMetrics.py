'''
Created on 09.03.2023

@author: vital
'''
import torch
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
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.BinaryClassificationMetricsPlots import BinaryClassificationMetricsPlots
from classifier.ModelEvaluationHelper import loadModel,\
    createTweetGroupsAndTrueClasses
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler

word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt", binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
model = loadModel(DataDirHelper().getDataDir()+"companyTweets\\model\\teslaCarSalesLSTM5\\tweetpredict_fold0.ckpt",word_vectors)
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsTeslaWithCarSales.csv')
#df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df) otherwise splits would be wrong when working with whole df
testSplitIndexes = np.load(DataDirHelper().getDataDir()+"companyTweets\\model\\teslaCarSalesLSTM5\\test_idx_fold0.npy")
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        5,
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

