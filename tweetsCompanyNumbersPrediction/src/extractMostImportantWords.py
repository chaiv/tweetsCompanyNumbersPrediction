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
from featureinterpretation.TokenAttributionStore import TokenAttributionStore

tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictionClassMapper = BINARY_0_1 
tokenAttributionStore = TokenAttributionStore()
predictor = Predictor(model,tokenizer,predictionClassMapper)
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
df.fillna('', inplace=True) #nan values in body columns
sentences = df["body"].tolist()
predictions = predictor.predictMultipleInChunks(sentences,chunkSize=10000)
df['predicted_class'] = predictions
correctly_predicted_df = df.loc[df['class'] == df['predicted_class']]
for index, row in correctly_predicted_df.iterrows():
    print(index)
    tweet_id = row["tweet_id"]
    tokens, attributions = predictor.calculateWordScoresOne(row["body"], predictionClassMapper.class_to_index(row['predicted_class']))
    tokenAttributionStore.add_data(tweet_id, tokens, attributions)
tokenAttributionsDf = tokenAttributionStore.to_dataframe()














