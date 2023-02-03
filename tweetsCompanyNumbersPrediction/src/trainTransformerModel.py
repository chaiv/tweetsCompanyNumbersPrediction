'''
Created on 03.02.2023

@author: vital
'''
from functools import partial
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from classifier.transformer.models import Transformer
from tweetpreprocess.DataDirHelper import DataDirHelper
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec

class Dataset(Dataset):
    def __init__(self, dataframe,tokenizer, textColumnName = "body" , classColumnName = "class"):
        self.dataframe = dataframe
        self.textColumnName = textColumnName
        self.classColumnName = classColumnName
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = str(self.dataframe.loc[idx, self.textColumnName])
        label = self.dataframe.loc[idx,  self.classColumnName]

        x = self.tokenizer.encode(text)
        y = label

        x = torch.tensor(x, dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    x_input, y_output = [], []
    for (x, y) in data_batch:
        x_input.append(x)
        y_output.append(y)
    x_input = pad_sequence(x_input, padding_value=pad_idx, batch_first=True)
    y_output = torch.tensor(y_output, dtype=torch.long)
    return x_input, y_output

if  __name__ == "__main__":
    batch_size = 64
    epochs = 10

    df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsAAPLFirst1000WithNumbers.csv')
    modelpath =  DataDirHelper().getDataDir()+ "companyTweets\TopicModelAAPLFirst1000"
    topicExtractor = TopicExtractor(TopicModelCreator().load(modelpath))
    tokenizer = TokenizerTop2Vec(topicExtractor.getWordIndexes())
    pad_token_idx = tokenizer.getPADTokenID()
    vocab_size = tokenizer.getVocabularyLength()

    train_val, test = train_test_split(df, random_state=1337, test_size=0.3)
    train, val = train_test_split(train_val, random_state=1337, test_size=0.3)

    train_data = Dataset(dataframe=train,tokenizer = tokenizer)
    val_data = Dataset(dataframe=val,tokenizer =tokenizer)
    test_data = Dataset(dataframe=test,tokenizer = tokenizer)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=pad_token_idx),
    )

    base_path = Path(__file__).parents[1]

    model = Transformer(lr=1e-4, n_outputs=3, vocab_size=vocab_size)

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
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

