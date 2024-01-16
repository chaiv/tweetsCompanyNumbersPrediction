from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.DataframeSplitter import DataframeSplitter

'''
Created on 03.02.2023

@author: vital
'''
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold,train_test_split
from gensim.models import KeyedVectors
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.LSTMNN import LSTMNN
from classifier.Trainer import Trainer
from classifier.TweetGroupDataset import TweetGroupDataset


torch.set_float32_matmul_precision('medium') #needed for quicker cuda 

if  __name__ == "__main__":
     
    #df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAAPLFirst1000WithNumbers.csv") 
    #word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAAPLFirst1000.txt", binary=False) 
    df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv")
    df.fillna('', inplace=True) #nan values in body columns 
    word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()
    
    model =LSTMNN(emb_size = 300,word_vectors = word_vectors)

    # model = Transformer(
    #     embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=emb_size,pad_token_id = textEncoder.getPADTokenID()),
    #     lr=1e-4, n_outputs=2, vocab_size=vocab_size,channels= 300
    #     ) #https://discuss.pytorch.org/t/solved-assertion-srcindex-srcselectdimsize-failed-on-gpu-for-torch-cat/1804/13
    
    splitter = DataframeSplitter()
    tweetSplits = splitter.getSplitIds(df, 5) #how many tweets should be trained as one sample
    kfold_splits = 3
    kfold_cross_val = KFold(n_splits=kfold_splits, shuffle=True, random_state=1337)
    for fold, (train_idx, test_idx) in enumerate(kfold_cross_val.split(tweetSplits)):
        np.save(DataDirHelper().getDataDir() + f'companyTweets\\model\\teslaCarSalesLSTM5\\test_idx_fold{fold}.npy', test_idx) #save test indexes for later classification metrics
        train_idx, val_idx = train_test_split(train_idx, random_state=1337, test_size=0.3)
        # print("Train classes",splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,train_idx))
        # print("Val classes", splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,val_idx))
        # print("Test classes", splitter.getClassCountsOfSplitsByIndexes(df,tweetSplits,test_idx))
        train_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes= train_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        val_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes = val_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        test_data = TweetGroupDataset(dataframe=df,splits = tweetSplits, splitIndexes = test_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        print("len(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))
        Trainer().train(
            batch_size=100, 
            epochs=10, 
            num_workers=8, 
            pad_token_idx=pad_token_idx, 
            model=model, 
            train_data=train_data, 
            val_data=val_data, 
            test_data=test_data, 
            loggerPath=DataDirHelper().getDataDir() + 'companyTweets\\modellogs', 
            loggerName="tweetpredict", 
            checkpointPath=DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5', 
            checkpointName=f"tweetpredict_fold{fold}"
            )
        
   

   
        
