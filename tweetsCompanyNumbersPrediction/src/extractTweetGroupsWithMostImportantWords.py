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
tweetGroups1Label = []
tweetGroups0Label = []

for tweetGroup in tweetGroups: 
    if tweetGroup.getLabel()==1: 
        tweetGroups1Label.append(tweetGroup)
    else: 
        tweetGroups0Label.append(tweetGroup)


observed_class=0
tweetGroupsAmount = 100
if observed_class==1:
    observedTweetGroups = tweetGroups1Label[:tweetGroupsAmount]
else: 
    observedTweetGroups = tweetGroups0Label[:tweetGroupsAmount]

predictor = Predictor(model,tokenizer ,textEncoder,BINARY_0_1,AttributionsCalculator(model,model.embedding))
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(observedTweetGroups,1000)
wordScoresWrappers = predictor.calculateWordScoresOfTweetGroupsInChunks(observedTweetGroups,observed_class=observed_class,chunkSize=100,n_steps=500,internal_batch_size = 100)

tweetGroupColumn = "tweet_group"
tokensColumn = "tokens_sorted"
scoresColumn = "scores_sorted"
labelColumn = "true_label"
lstmLabel = "lstm_label"
chatGptLabel = "chatgpt_label"
chatGpttokens = "chatgpt_tokens"

tweetGroupLists = []
labelLists = []
tokensSortedByScoreLists = []
scoreSortedByScoreLists = []


sorter = TokenScoresSort()
for wordScoreWrapper in wordScoresWrappers:
    labelLists.append(wordScoreWrapper.getTweetGroup().getLabel())
    tweetGroupLists.append(";".join(wordScoreWrapper.getTweetGroup().getSentences()))
    sorted_tokens, sorted_scores = sorter.getSortedTokensAndScoresAscFromListOfLists(wordScoreWrapper.getTokens(), wordScoreWrapper.getAttributions())
    tokensSortedByScoreLists.append(sorted_tokens)
    scoreSortedByScoreLists.append(sorted_scores)


tweetGroupsWithMostImportantWordsDf = pd.DataFrame({
    tweetGroupColumn: tweetGroupLists,
    labelColumn: labelLists,
    tokensColumn : tokensSortedByScoreLists,
    #scoresColumn : scoreSortedByScoreLists ,
    lstmLabel : prediction_classes,
    chatGptLabel : [None] * len(labelLists),
    chatGpttokens: [None] * len(labelLists)
})

tweetGroupsWithMostImportantWordsDf.to_csv(predictionModelPath.getModelPath()+"\\tweetGroups_with_important_words_label_"+str(observed_class)+".csv")

