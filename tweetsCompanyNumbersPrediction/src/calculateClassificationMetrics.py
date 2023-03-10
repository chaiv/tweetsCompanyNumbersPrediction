'''
Created on 09.03.2023

@author: vital
'''
import torch
import pandas as pd
from classifier.transformer.models import Transformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
from nlpvectors.VocabularyIDEncoder import VocabularyIDEncoder
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from nlpvectors.TweetTokenizer import TweetTokenizer
from classifier.ClassificationMetrics import ClassificationMetrics

encoder = VocabularyIDEncoder(DataDirHelper().getDataDir()+ "companyTweets\VocabularyTesla.json")
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength())
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\teslaTweetPredictCarSales.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictionClassMapper = BINARY_0_1 
predictor = Predictor(model,TweetTokenizer(DefaultWordFilter()),encoder,predictionClassMapper,AttributionsCalculator(model))
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsTeslaWithCarSales.csv')
df.fillna('', inplace=True) #nan values in body columns
df = df.sample(n=100000)
sentences = df["body"].tolist()
true_classes = df["class"].tolist()
predictions = predictor.predictMultipleInChunks(sentences,chunkSize=10000)
metrics = ClassificationMetrics() 
print(metrics.calculate_metrics(true_classes, predictions, 0))
print(metrics.calculate_metrics(true_classes, predictions, 1))


