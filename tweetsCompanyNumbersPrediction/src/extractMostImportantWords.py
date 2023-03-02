'''
Created on 05.02.2023

@author: vital
'''
import torch
import pandas as pd
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from featureinterpretation.ImportantWordsStore import ImportantWordStore
from featureinterpretation.AttributionsCalculator import AttributionsCalculator

tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictionClassMapper = BINARY_0_1 
predictor = Predictor(model,tokenizer,predictionClassMapper,AttributionsCalculator(model))
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df = df.head(2)
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
importantWordsDf.to_csv(DataDirHelper().getDataDir()+"companyTweets\\importantWordsClass1Amazon.csv")


















