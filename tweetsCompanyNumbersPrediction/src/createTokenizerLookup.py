'''
Created on 19.11.2023

@author: vital
'''
import json 
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.TokenizerWordsLookup import TokenizedWordsLookup,\
    createLookUpDictionary
from PredictionModelPath import APPLE__EPS_10_LSTM_MULTI_CLASS
predictionModelPath = APPLE__EPS_10_LSTM_MULTI_CLASS
df = pd.read_csv(predictionModelPath.getDataframePath())
df.fillna('', inplace=True) #nan values in body columns
tweetSentences = df["body"].tolist()
lookupDict = createLookUpDictionary(tweetSentences)
with open(predictionModelPath.getModelPath()+"\\tokensDictionary.json", 'w') as f:
    f.write(json.dumps(lookupDict))
