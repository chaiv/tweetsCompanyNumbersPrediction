'''
Created on 08.02.2023

@author: vital
'''
import pandas as pd
import torch
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerEncoder
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
if  __name__ == "__main__":
    tokenizer = TokenizerEncoder(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
    vocab_size = tokenizer.getVocabularyLength()
    model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
    checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
    model.load_state_dict(checkpoint['state_dict'])
    df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
    df = df[df["class"]==1.0]
    print(df)
    Predictor(model,textEncoder).test(df)