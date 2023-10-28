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
from classifier.BinaryClassificationMetrics import BinaryClassificationMetrics
from gensim.models import KeyedVectors
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset

word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
textEncoder = WordVectorsIDEncoder(word_vectors)
tokenizer = TweetTokenizer(DefaultWordFilter())
checkpointName = "\\amazonRevenueLSTMN5\\tweetpredict_fold0.ckpt"
model = LSTMNN(300,word_vectors)
# model = Transformer(
#         embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=300,pad_token_id = encoder.getPADTokenID()),
#         lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength(),channels= 300
#         )
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\tweetpredict_fold2.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictor = Predictor(model,tokenizer,textEncoder ,BINARY_0_1 ,None)
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df.fillna('', inplace=True) #nan values in body columns
splits = DataframeSplitter().getSplitIds(df,5)
testSplitIndexes = np.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonRevenueLSTMN5\\test_idx_fold2.npy")
test_dataset = TweetGroupDataset(dataframe=df,splits = splits, splitIndexes= testSplitIndexes, tokenizer=tokenizer, textEncoder=textEncoder)
tweetGroups = []
true_classes = []
for i in range(len(test_dataset)):
    tweetGroup = test_dataset.getAsTweetGroup(i)
    tweetGroups.append(tweetGroup)
    true_classes.append(tweetGroup.getLabel())
prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
metrics = BinaryClassificationMetrics() 
print(metrics.classification_report(true_classes, prediction_classes))


