'''
Created on 09.03.2023

@author: vital
'''
import pandas as pd
import numpy as np
from nlpvectors.DataframeSplitter import DataframeSplitter
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from classifier.ClassificationMetrics import ClassificationMetrics
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.ModelEvaluationHelper import loadModel
from classifier.TweetGroupDataset import TweetGroupDataset
from collections import Counter
from tweetpreprocess.EqualClassSampler import EqualClassSampler

from tweetpreprocess.LoadTweetDataframe import LoadTweetDataframe
from classifier.PredictionClassMapper import PredictionClassMapper
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_MULTI_CLASS,\
    AMAZON_REVENUE_20_LSTM_MULTI_CLASS, APPLE__EPS_10_LSTM_MULTI_CLASS,\
    TESLA_CAR_SALES_10_LSTM_MULTI_CLASS

predictionModelPath = APPLE__EPS_10_LSTM_MULTI_CLASS
predictionClassMapper = predictionModelPath.getPredictionClassMapper()
fold = 0
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
modelPath = predictionModelPath.getModelPath()+"\\tweetpredict_fold"+str(fold)+".ckpt"
model = loadModel(modelPath,word_vectors,num_classes=predictionClassMapper.get_number_of_classes())
# Read raw dataframe WITHOUT EqualClassSampler (consistent with training)
df = pd.read_csv(predictionModelPath.getDataframePath())
df.fillna('', inplace=True)
testIdxPath = predictionModelPath.getModelPath()+'\\test_idx_fold'+str(fold)+'.npy'
testSplitIndexes = np.load(testIdxPath)

# Apply same time-based sorting as in training
splitter = DataframeSplitter()
splits = splitter.getSplitIds(df, predictionModelPath.getTweetGroupSize())
postTSPColumn = "post_date"
df[postTSPColumn] = pd.to_datetime(df[postTSPColumn])
tweet_id_to_date = dict(zip(df["tweet_id"], df[postTSPColumn]))
split_dates = [min(tweet_id_to_date[tid] for tid in split) for split in splits]
sorted_indices = np.argsort(split_dates)
splits = [splits[i] for i in sorted_indices]

df.fillna('', inplace=True)
test_dataset = TweetGroupDataset(dataframe=df, splits=splits, splitIndexes=testSplitIndexes, tokenizer=tokenizer, textEncoder=textEncoder)
tweetGroups = []
trueClasses = []
for i in range(len(test_dataset)):
    tweetGroup = test_dataset.getAsTweetGroup(i)
    tweetGroups.append(tweetGroup)
    trueClasses.append(tweetGroup.getLabel())
    print("created tweet group "+str(i))
predictor = Predictor(model,tokenizer,textEncoder ,predictionClassMapper ,None)
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
print("true_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
print("prediction_classes counts ",', '.join(f"{item}: {count}" for item, count in Counter(prediction_classes).items()))
metrics = ClassificationMetrics()
print(metrics.classification_report(trueClasses, prediction_classes))
print("MCC "+str(metrics.calculate_mcc(trueClasses, prediction_classes)))
#BinaryClassificationMetricsPlots(metrics).createRocAUCAndPrAucPlots(trueClasses, prediction_classes)

