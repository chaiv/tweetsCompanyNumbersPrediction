'''
Created on 05.02.2023

@author: vital
'''
import torch
import pandas as pd
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

word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
encoder = WordVectorsIDEncoder(word_vectors)
checkpointName = "lstmAmazonTweetPredict.ckpt"
model = LSTMNN(300,word_vectors)
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\"+checkpointName)
model.load_state_dict(checkpoint['state_dict'])
model.train()
predictionClassMapper = BINARY_0_1 
predictor = Predictor(model,TweetTokenizer(DefaultWordFilter()),encoder,predictionClassMapper,AttributionsCalculator(model,model.embedding))
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df = df.head(10000)
df.fillna('', inplace=True) #nan values in body columns
sentence_ids = df["tweet_id"].tolist()
sentences = df["body"].tolist()
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


















