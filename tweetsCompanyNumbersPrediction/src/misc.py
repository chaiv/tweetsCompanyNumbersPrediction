'''
Created on 06.11.2023

@author: vital
'''
import torch
import pandas as pd
import numpy as np
from classifier.transformer.models import Transformer
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
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.ModelEvaluationHelper import createTweetGroupsAndTrueClasses,\
    loadModel, createTweetGroupsAndTrueClassesWithoutSplitIndexes
from nlpvectors.TweetGroup import createTweetGroup
    
word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
model = loadModel(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetpredict_fold1.ckpt",word_vectors,evalMode=False)
sentences = ['For Options$AAPL$PFE$ORCL$MRK$NLY$COP$COH$DVN$NFX$TTWO$AMZNNice bottom', 'Stocks Trending Now:  $LL $LGND $BIOC $CSCO $ICA $USLV $ZIOP $CLDX $MHYS $ODP $JUNO $AMZN ~', 'A few more of interest: $AMBA $ADXS $AAL $AMGN $AMZN $ARIA $ARRY $ASHR ', 'Shares of Zulily down another 6 percent on Friday, slumping to all-time low () $ZU $AMZN $NILE ', '$XLI Investor Opinions Updated Friday, March 13, 2015 7:03:23 PM $INTC $AMZN $MCP $CSCO ']
sentenceIds = [576527445806870528, 576529132651196416, 576531462842949632, 576534180621729792, 576534748559863808]
label = 1
tweetGroup = createTweetGroup(tokenizer,textEncoder , sentences,sentenceIds,label)
predictor = Predictor(model,tokenizer ,textEncoder,BINARY_0_1,AttributionsCalculator(model,model.embedding))
wordScoresWrappers = predictor.calculateWordScoresOfTweetGroupsInChunks([tweetGroup],observed_class=1,chunkSize=100,n_steps=500,internal_batch_size = 100)
print(wordScoresWrappers)

