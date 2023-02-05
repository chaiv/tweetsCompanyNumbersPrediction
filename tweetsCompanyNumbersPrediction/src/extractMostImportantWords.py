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
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetpredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#text = "$AMZN - 21st Century Fox Earnings: An EPS Beat, but Revenue Misses Expectations" 
text = "S&P100 #Stocks Performance $HD $LOW $SBUX $TGT $DVN $IBM $AMZN $F $APA $GM $MS $HAL $DIS $MCD $BMY $XOM  more@"
x_Tokenids = tokenizer.encode(text)
x_Tokenids  += [tokenizer.getPADTokenID()]*(256 - len(x_Tokenids)) #padding
with torch.no_grad():
    pred = model(torch.tensor(x_Tokenids , dtype=torch.long))
    pred = torch.sigmoid(pred)
    pred = torch.round(pred)
    print(pred)
