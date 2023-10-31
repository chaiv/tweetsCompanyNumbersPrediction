'''
Created on 05.02.2023

@author: vital
'''
import torch
import pandas as pd
import numpy as np
from classifier.transformer.models import Transformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from featureinterpretation.ImportantWordsStore import ImportantWordStore
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
from nlpvectors.VocabularyIDEncoder import VocabularyIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset
from calculateClassificationMetrics import loadModel,\
    createTweetGroupsAndTrueClasses




word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
model = loadModel(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetpredict_fold1.ckpt",word_vectors)
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
testSplitIndexes = np.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\test_idx_fold1.npy")
tweetGroups,trueClasses = createTweetGroupsAndTrueClasses(
        df,
        5,
        testSplitIndexes,
        tokenizer,
        textEncoder
        )
predictor = Predictor(model,tokenizer ,textEncoder,BINARY_0_1,AttributionsCalculator(model,model.embedding))


predictions = predictor.predictMultipleInChunks(sentences,chunkSize=10000)
observed_class = 1
token_indexes, tokens, attributions =  predictor.calculateWordScoresInChunks(sentences, observed_class, chunk_size=200, n_steps=10, internal_batch_size=200)
importantWordsDf = ImportantWordStore(
                {
                "id" : sentence_ids,
                "token_index" : token_indexes,
                "token" : tokens,
                "attribution" : attributions,
                "prediction" : predictions
                }
                ).to_dataframe()
importantWordsDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\importantWordsClass"+str(observed_class)+"Amazon.csv")


















