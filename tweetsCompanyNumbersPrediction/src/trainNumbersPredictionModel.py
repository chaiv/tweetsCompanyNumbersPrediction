from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.DataframeSplitter import DataframeSplitter

'''
Created on 03.02.2023

@author: vital
'''
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.LSTMNN import LSTMNN
from classifier.Trainer import Trainer
from classifier.TweetGroupDataset import TweetGroupDataset
from tweetpreprocess.EqualClassSampler import EqualClassSampler

from tweetpreprocess.LoadTweetDataframe import LoadTweetDataframe
from classifier.CreateClassifierModel import CreateClassifierModel
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_MULTI_CLASS,\
    AMAZON_REVENUE_20_LSTM_MULTI_CLASS, TESLA_CAR_SALES_10_LSTM_MULTI_CLASS, \
    APPLE__EPS_10_LSTM_MULTI_CLASS


torch.set_float32_matmul_precision('medium') #needed for quicker cuda 

if  __name__ == "__main__":
    predictionModelPath =  APPLE__EPS_10_LSTM_MULTI_CLASS

    df = pd.read_csv(predictionModelPath.getDataframePath()) 
    df.fillna('', inplace=True) #nan values in body columns 
    df = LoadTweetDataframe(predictionModelPath).readDataframe()
    print(TweetDataframeExplore(df).getClassDistribution())
    word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()
    

    # model = Transformer(
    #     embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=emb_size,pad_token_id = textEncoder.getPADTokenID()),
    #     lr=1e-4, n_outputs=2, vocab_size=vocab_size,channels= 300
    #     ) #https://discuss.pytorch.org/t/solved-assertion-srcindex-srcselectdimsize-failed-on-gpu-for-torch-cat/1804/13
    
    splitter = DataframeSplitter()
    tweetSplits = splitter.getSplitIds(df, predictionModelPath.getTweetGroupSize()) #how many tweets should be trained as one sample

    postTSPColumn = "post_date"
    df[postTSPColumn] = pd.to_datetime(df[postTSPColumn])
    split_dates = []
    for split in tweetSplits:
        earliest_date = df[df["tweet_id"].isin(split)][postTSPColumn].min()
        split_dates.append(earliest_date)
    sorted_indices = np.argsort(split_dates)
    tweetSplits = [tweetSplits[i] for i in sorted_indices]

    # Get class label for each sorted split
    split_labels = []
    for split in tweetSplits:
        label = df[df["tweet_id"].isin(split)]["class"].iloc[0]
        split_labels.append(label)
    split_labels = np.array(split_labels)

    # Time-based expanding window cross-validation with stratified temporal split
    n = len(tweetSplits)
    kfold_splits = 2

    # Group indices by class, preserving temporal order within each class
    unique_classes = np.unique(split_labels)
    class_indices = {c: np.where(split_labels == c)[0] for c in unique_classes}

    # Calculate fold steps per class
    class_test_sizes = {c: int(len(idx) * 0.3) for c, idx in class_indices.items()}
    class_remaining = {c: len(idx) - class_test_sizes[c] for c, idx in class_indices.items()}
    class_fold_steps = {c: class_remaining[c] // kfold_splits for c in unique_classes}

    for fold in range(kfold_splits):
        train_val_all = []
        test_all = []
        for c in unique_classes:
            c_indices = class_indices[c]
            c_n = len(c_indices)
            # Expanding window per class
            c_train_val_end = class_test_sizes[c] + class_fold_steps[c] * (fold + 1)
            if fold == kfold_splits - 1:
                c_train_val_end = c_n
            c_train_val_count = int(c_train_val_end * 0.7)
            # First 70% of window = train/val, last 30% = test (temporally ordered)
            train_val_all.extend(c_indices[:c_train_val_count].tolist())
            test_all.extend(c_indices[c_train_val_count:c_train_val_end].tolist())

        train_val_idx = np.array(train_val_all)
        test_idx = np.array(test_all)
        testIdxPath = predictionModelPath.getModelPath()+f"\\test_idx_fold{fold}.npy"
        np.save(testIdxPath, test_idx)
        # Stratified train/val split
        train_val_labels = split_labels[train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, random_state=1337, test_size=0.3, stratify=train_val_labels)
        print(f"=== Fold {fold} ===")
        print("Train classes",splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,train_idx))
        print("Val classes", splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,val_idx))
        print("Test classes", splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,test_idx))
        train_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes= train_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        val_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes = val_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        test_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes = test_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        print("len(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))
        # Re-create model for each fold to avoid weight leakage between folds
        model = CreateClassifierModel(word_vectors=word_vectors, num_classes=predictionModelPath.getPredictionClassMapper().get_number_of_classes()).createModel()
        Trainer().train(
            batch_size=100, 
            epochs=10, 
            num_workers=2,
            pad_token_idx=pad_token_idx,
            model=model, 
            train_data=train_data, 
            val_data=val_data, 
            test_data=test_data, 
            loggerPath=DataDirHelper().getDataDir() + 'companyTweets\\modellogs', 
            loggerName="tweetpredict", 
            checkpointPath=predictionModelPath.getModelPath(), 
            checkpointName=f"tweetpredict_fold{fold}"
            )
        
   

   
        
