'''
Created on 05.02.2023

@author: vital
'''
import torch
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tweetpreprocess.DataDirHelper import DataDirHelper

tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
model.load_state_dict(torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetpredict.ckpt"),strict=False)
model.eval()
x_Tokenids = tokenizer.encode("S&P100 #Stocks Performance $HD $LOW $SBUX $TGT $DVN $IBM $AMZN $F $APA $GM $MS $HAL $DIS $MCD $BMY $XOM  more@")
model(
    torch.tensor(x_Tokenids , dtype=torch.long)
    )