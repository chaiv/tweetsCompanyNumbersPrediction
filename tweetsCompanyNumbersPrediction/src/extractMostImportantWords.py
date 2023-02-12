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

tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

text1 = "$AMZN - 21st Century Fox Earnings: An EPS Beat, but Revenue Misses Expectations" 
text2 = "S&P100 #Stocks Performance $HD $LOW $SBUX $TGT $DVN $IBM $AMZN $F $APA $GM $MS $HAL $DIS $MCD $BMY $XOM  more@"
text3 = "Amazon stock price target raised to $1,360 at Instinet, making it highest on FactSet  $amzn"
predictor = Predictor(model,tokenizer)
print(predictor.predictOne(text3))
print(predictor.predictMultiple([text1,text2,text3]))

#print(predictor(text1,1,tokenizer,model)) 
df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonTweetsWithNumbers.csv')
sentences = df["body"].tolist()
predictions = predictor.predictMultiple(sentences)
df['predicted_class'] = predictions
correctly_predicted_df = df.loc[df['class'] == df['predicted_class']]














