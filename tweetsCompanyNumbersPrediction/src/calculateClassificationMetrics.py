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
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.transformer.Word2VecTransformerEmbedding import Word2VecTransformerEmbedding
from classifier.FeedForwardNN import FeedForwardNN
from classifier.LSTMNN import LSTMNN

word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
encoder = WordVectorsIDEncoder(word_vectors)
checkpointName = "lstmAmazonTweetPredict.ckpt"
model = LSTMNN(300,word_vectors)
# model = Transformer(
#         embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=300,pad_token_id = encoder.getPADTokenID()),
#         lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength(),channels= 300
#         )
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\"+checkpointName)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictionClassMapper = BINARY_0_1 
predictor = Predictor(model,TweetTokenizer(DefaultWordFilter()),encoder,predictionClassMapper,None)
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df.fillna('', inplace=True) #nan values in body columns
df = df.sample(n=10000)
sentences = df["body"].tolist()
true_classes = df["class"].tolist()
predictions = predictor.predictMultipleInChunks(sentences,chunkSize=1000)
metrics = ClassificationMetrics() 
print(metrics.classification_report(true_classes, predictions))


