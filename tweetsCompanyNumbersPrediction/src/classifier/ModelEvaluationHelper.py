'''
Created on 04.11.2023

@author: vital
'''
import torch
import pandas as pd
import numpy as np
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.LSTMNN import LSTMNN, LAST_HIDDEN_STATE_POOLING
from nlpvectors.DataframeSplitter import DataframeSplitter
from classifier.TweetGroupDataset import TweetGroupDataset
from classifier.CreateClassifierModel import CreateClassifierModel


def _warnIfNewerVersionedCheckpointExists(path):
    """ModelCheckpoint appends -v1, -v2 ... instead of overwriting, so after a re-run the plain
    filename is the OLD model. Loading it with strict=False raises nothing, which silently scores
    the previous run; the training scripts now use the path returned by Trainer.train."""
    import glob
    from os import path as ospath
    if not ospath.exists(path):
        return
    stem, ext = ospath.splitext(path)
    newer = [p for p in glob.glob(stem + "-v*" + ext) if ospath.getmtime(p) > ospath.getmtime(path)]
    if newer:
        print("WARNING: newer versioned checkpoint(s) exist next to %s: %s. You are probably loading "
              "the model of an older run." % (path, sorted(newer)))


def loadModel(path,wordVectors,num_classes=2,evalMode=True,pooling=LAST_HIDDEN_STATE_POOLING,
              padTokenIdx=None,device=None,lstmDropout=0.0):
    # pooling has to match the value the checkpoint was trained with, it changes how the token
    # sequence is reduced and therefore what the trained weights mean.
    model = CreateClassifierModel(word_vectors = wordVectors,num_classes = num_classes,
                                  pooling = pooling, padTokenIdx = padTokenIdx,
                                  lstmDropout = lstmDropout).createModel()
    # model = Transformer(
    #         embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=300,pad_token_id = encoder.getPADTokenID()),
    #         lr=1e-4, n_outputs=2, vocab_size=encoder.getVocabularyLength(),channels= 300
    #         )
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model = model.to(device)
    _warnIfNewerVersionedCheckpointExists(path)
    checkpoint = torch.load(path, map_location=device)
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("WARNING: checkpoint/model mismatch. Missing keys: %s; unexpected keys: %s" %
              (incompatible.missing_keys, incompatible.unexpected_keys))
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
