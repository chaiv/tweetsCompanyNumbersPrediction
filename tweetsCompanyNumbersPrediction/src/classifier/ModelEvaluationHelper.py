'''
Created on 04.11.2023

@author: vital
'''
import torch
import pandas as pd
import numpy as np
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.LSTMNN import LSTMNN
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset


def loadModel(path,wordVectors,evalMode=True):
    model = LSTMNN(300,wordVectors)
    # model = Transformer(
    #         embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=300,pad_token_id = encoder.getPADTokenID()),
    #         lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength(),channels= 300
    #         )
    model = model.to(torch.device("cuda:0"))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    if(evalMode):
        model.eval()
    return model
    


def createTweetGroupsAndTrueClasses(
        tweetDf,
        splitNumber,
        splitIndexes,
        tokenizer,
        textEncoder
        ):
    tweetDf.fillna('', inplace=True) #nan values in body columns
    splits = DataframeSplitter().getSplitIds(tweetDf,splitNumber)
    test_dataset = TweetGroupDataset(dataframe=tweetDf,splits = splits, splitIndexes= splitIndexes, tokenizer=tokenizer, textEncoder=textEncoder)
    tweetGroups = []
    trueClasses = []
    for i in range(len(test_dataset)):
        tweetGroup = test_dataset.getAsTweetGroup(i)
        tweetGroups.append(tweetGroup)
        trueClasses.append(tweetGroup.getLabel())
        print("created tweet group "+str(i))
    return tweetGroups,trueClasses
    