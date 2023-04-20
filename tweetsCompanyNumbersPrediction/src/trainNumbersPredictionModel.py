
'''
Created on 03.02.2023

@author: vital
'''
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold,train_test_split

from gensim.models import KeyedVectors
from classifier.transformer.models import Transformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from classifier.transformer.DatasetUtils import Dataset,createDataloader
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.transformer.Word2VecTransformerEmbedding import Word2VecTransformerEmbedding
from classifier.FeedForwardNN import FeedForwardNN
from classifier.LSTMNN import LSTMNN
from classifier.Trainer import Trainer

torch.set_float32_matmul_precision('medium') #needed for quicker cuda 

if  __name__ == "__main__":
    batch_size = 2048
    epochs = 10
    num_workers = 8
    emb_size = 300
     
    df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv") 
    word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    
    #df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
    
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()
    
    model =LSTMNN(emb_size,word_vectors)

    # model = Transformer(
    #     embeddings= Word2VecTransformerEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=emb_size,pad_token_id = textEncoder.getPADTokenID()),
    #     lr=1e-4, n_outputs=2, vocab_size=vocab_size,channels= 300
    #     ) #https://discuss.pytorch.org/t/solved-assertion-srcindex-srcselectdimsize-failed-on-gpu-for-torch-cat/1804/13
    
    kfold_cross_val = KFold(n_splits=10, shuffle=True, random_state=1337)
    for fold, (train_idx, test_idx) in enumerate(kfold_cross_val.split(df)):
        train_idx, val_idx = train_test_split(train_idx, random_state=1337, test_size=0.3)
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
    
        print("Train classes",TweetDataframeExplore(train_df).getClassDistribution())
        print("Val classes",TweetDataframeExplore(val_df).getClassDistribution())
        print("Test classes",TweetDataframeExplore(test_df).getClassDistribution())
        
        train_data = Dataset(dataframe=train_df,tokenizer = tokenizer, textEncoder = textEncoder)
        val_data = Dataset(dataframe=val_df,tokenizer = tokenizer,textEncoder =textEncoder)
        test_data = Dataset(dataframe=test_df,tokenizer = tokenizer,textEncoder = textEncoder)
    
        print("len(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))
        
        Trainer().train(
            batch_size=batch_size, 
            epochs=epochs, 
            num_workers=num_workers, 
            pad_token_idx=pad_token_idx, 
            model=model, 
            train_data=train_data,
            val_data=val_data, 
            test_data=test_data,
            loggerPath = DataDirHelper().getDataDir() + 'companyTweets\\modellogs', 
            loggerName = "tweetpredict", 
            checkpointPath= DataDirHelper().getDataDir() + 'companyTweets\\model', 
            checkpointName = f"tweetpredict_fold{fold}"
            )

