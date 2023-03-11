'''
Created on 03.02.2023

@author: vital
'''
from functools import partial
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
from classifier.transformer.models import Transformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from classifier.transformer.DatasetUtils import generate_batch, Dataset
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.transformer.Word2VecTokenEmbedding import Word2VecTokenEmbedding

torch.set_float32_matmul_precision('medium')

if  __name__ == "__main__":
    #batch_size = 2048
    batch_size = 256
    epochs = 10
    num_workers = 8
    emb_size = 300
     
    df = pd.read_csv(DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv") 
    word_vectors = KeyedVectors.load_word2vec_format(DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt", binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    embeddings = Word2VecTokenEmbedding(word_vectors =  torch.tensor(word_vectors.vectors), emb_size=emb_size,pad_token_id = textEncoder.getPADTokenID())
    
    df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
    
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()

    train_val, test = train_test_split(df, random_state=1337, test_size=0.3)
    train, val = train_test_split(train_val, random_state=1337, test_size=0.3)
    
    print(TweetDataframeExplore(train).getClassDistribution())
    
    train_data = Dataset(dataframe=train,tokenizer = tokenizer, textEncoder = textEncoder)
    val_data = Dataset(dataframe=val,tokenizer = tokenizer,textEncoder =textEncoder)
    test_data = Dataset(dataframe=test,tokenizer = tokenizer,textEncoder = textEncoder)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )

    model = Transformer(
        embeddings= embeddings,
        lr=1e-4, n_outputs=2, vocab_size=vocab_size,channels= 300
        ) #https://discuss.pytorch.org/t/solved-assertion-srcindex-srcselectdimsize-failed-on-gpu-for-torch-cat/1804/13

    logger = TensorBoardLogger(
        save_dir=DataDirHelper().getDataDir()+ 'companyTweets\\modellogs',
        name="tweetpredict",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=DataDirHelper().getDataDir()+ 'companyTweets\\model',
        filename="tweetpredict",
        save_weights_only=True,
    )

    early_stopping = EarlyStopping(monitor="valid_loss", mode="min", patience=10)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

