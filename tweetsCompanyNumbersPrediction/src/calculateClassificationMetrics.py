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
from classifier.ClassificationMetrics import ClassificationMetrics
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter

word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
encoder = WordVectorsIDEncoder(word_vectors)
checkpointName = "\\amazonRevenueLSTMN5\\tweetpredict_fold0.ckpt"
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


testsamples = DataframeSplitter().splitDfByNSamplesForClass(df.iloc[np.load(DataDirHelper().getDataDir()+"companyTweets\\model\\test_idx_fold0.npy")],5, 'class')
sentenceWrappers = df["body"].tolist()
true_classes = [testsample["class"] for testsample in testsamples]
predictions = predictor.predictMultipleInChunks(sentences,chunkSize=1000)
metrics = ClassificationMetrics() 
print(metrics.classification_report(true_classes, predictions))


