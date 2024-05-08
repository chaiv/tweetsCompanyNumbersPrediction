'''
Created on 06.11.2023

@author: vital
'''
import torch
import pandas as pd
import numpy as np
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from featureinterpretation.ImportantWordsStore import ImportantWordStore,\
    createImportantWordStore
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
from nlpvectors.VocabularyIDEncoder import VocabularyIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.ModelEvaluationHelper import createTweetGroupsAndTrueClasses,\
    loadModel, createTweetGroupsAndTrueClassesWithoutSplitIndexes
from nlpvectors.TweetGroup import createTweetGroup
from PredictionModelPath import AMAZON_REVENUE_5
from featureinterpretation.TokenScoresSort import TokenScoresSort
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup

predictionModelPath =  AMAZON_REVENUE_5
    
word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
model = loadModel(predictionModelPath.getModelPath()+"\\tweetpredict_fold1.ckpt",word_vectors,evalMode=False)

combinedSentences = "$TRIP Forum Updates Available: Check Them Out: $T $VWO $PFE $AMZN; $QCOM Price Movement Summary for Thursday, April 2, 2015 10:07:16 PM $MSFT $CVX $DWT $AMZN; Not endorsing @MarketMaverick from @CNBCMarketWatch, always negative. Glad I ignored your suggestions, Mark, and invested in $amzn when it was below $350 #shiftfrombearishbullish!; 4/2/15 - US: $DIA down 0.33%, $XLK down 0.50%, $VTI up 0.05% RED: $MSFT $GOOGL $AMZN $PYPL $DIS $NVDA $BABA $SQ $TWTR $SNAP $FB $CRM BLUE: $LYFT $NFLX; $AG Community Stock Update Thursday, April 2, 2015 11:07:16 PM $INTC $DELL $AMZN $CGC"

# sentences = [
#     '$AMZN acquisitions & #onlineactivity could be compromised - @Amazon refuses to implement #SecureWeb on all sites #privacyconcerns', 
#     '$OIL Company Profile Updated Sunday, April 5, 2015 7:15:51 PM $PFE $YUM $AMZN $GLUU', 
#     'Cash Flow Analysis for Retailers 1. $WMT 2. $AMZN 3. $TGT Complete Details: #MarketTrends', 
#     '$AMZN Recent News Sunday, April 5, 2015 7:08:11 PM $GILD $BMY $IYE $SLX', 
#     '$MTD Current Updates Sunday, April 5, 2015 7:02:20 PM $GSK $SPY $AMZN $JNK'
#     ]

sentences = combinedSentences.split(";")

sentenceIds = [1, 2, 3, 4, 5]
label = 1
tweetGroup = createTweetGroup(tokenizer,textEncoder , sentences,sentenceIds,label)
predictor = Predictor(model,tokenizer ,textEncoder,BINARY_0_1,AttributionsCalculator(model,model.embedding))
wordScoresWrappers = predictor.calculateWordScoresOfTweetGroupsInChunks([tweetGroup],observed_class=0,chunkSize=100,n_steps=500,internal_batch_size = 100)
sorter = TokenScoresSort()
sorted_tokens, sorted_scores = sorter.getSortedTokensAndScoresAscFromListOfLists(wordScoresWrappers[0].getTokens(),wordScoresWrappers[0].getAttributions())
tokenizerLookUp = TokenizedWordsLookup(predictionModelPath.getModelPath()+"\\tokensDictionary.json")
print(predictor.predictMultipleAsTweetGroups([tweetGroup]))
print(sorter.getAsSingleList(tokenizerLookUp.getOriginalWords(sorted_tokens), sorted_scores))


