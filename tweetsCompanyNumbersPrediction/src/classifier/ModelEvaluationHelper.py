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
from classifier.CreateClassifierModel import CreateClassifierModel


def loadModel(path,wordVectors,num_classes=2,evalMode=True):
    model = CreateClassifierModel(word_vectors = wordVectors,num_classes = num_classes).createModel()
    # model = Transformer(
    #         embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=300,pad_token_id = encoder.getPADTokenID()),
    #         lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength(),channels= 300
    #         )
    model = model.to(torch.device("cuda:0"))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if(evalMode):
        model.eval()
    return model


def _sortSplitsTemporally(splits, tweetDf):
    """Sort splits by earliest tweet date (temporal order), matching training scripts."""
    postTSPColumn = "post_date"
    tweetDf[postTSPColumn] = pd.to_datetime(tweetDf[postTSPColumn])
    tweet_id_to_date = dict(zip(tweetDf["tweet_id"], tweetDf[postTSPColumn]))
    split_dates = [min(tweet_id_to_date[tid] for tid in split) for split in splits]
    sorted_indices = np.argsort(split_dates)
    return [splits[i] for i in sorted_indices]


def createTweetGroupsAndTrueClassesWithoutSplitIndexes(
         tweetDf,
        splitNumber,
        tokenizer,
        textEncoder,
        sortTemporally=False
        ):
        tweetDf.fillna('', inplace=True) #nan values in body columns
        splits = DataframeSplitter().getSplitIds(tweetDf,splitNumber)
        if sortTemporally:
            splits = _sortSplitsTemporally(splits, tweetDf)
        test_dataset = TweetGroupDataset(dataframe=tweetDf,splits = splits, splitIndexes= [i for i in range(0,len(splits))], tokenizer=tokenizer, textEncoder=textEncoder)
        tweetGroups = []
        trueClasses = []
        for i in range(len(test_dataset)):
            tweetGroup = test_dataset.getAsTweetGroup(i)
            tweetGroups.append(tweetGroup)
            trueClasses.append(tweetGroup.getLabel())
            print("created tweet group "+str(i))
        return tweetGroups,trueClasses
    

def createTweetGroupsAndTrueClasses(
        tweetDf,
        splitNumber,
        splitIndexes,
        tokenizer,
        textEncoder,
        sortTemporally=False
        ):
    tweetDf.fillna('', inplace=True) #nan values in body columns
    splits = DataframeSplitter().getSplitIds(tweetDf,splitNumber)
    if sortTemporally:
        splits = _sortSplitsTemporally(splits, tweetDf)
    test_dataset = TweetGroupDataset(dataframe=tweetDf,splits = splits, splitIndexes= splitIndexes, tokenizer=tokenizer, textEncoder=textEncoder)
    tweetGroups = []
    trueClasses = []
    for i in range(len(test_dataset)):
        tweetGroup = test_dataset.getAsTweetGroup(i)
        tweetGroups.append(tweetGroup)
        trueClasses.append(tweetGroup.getLabel())
        print("created tweet group "+str(i))
    return tweetGroups,trueClasses

    