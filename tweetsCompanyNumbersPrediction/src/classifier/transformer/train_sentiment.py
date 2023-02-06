import argparse
import os
import random
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
from classifier.transformer.nlp_utils import MAX_LEN, PAD_IDX, VOCAB_SIZE, tokenize
from classifier.transformer.predict_utils import attribution_fun,attribution_to_html
from tweetpreprocess.DataDirHelper import DataDirHelper

class Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.rating_to_indexes = {
            i: dataframe.index[dataframe["star_rating"] == i].tolist()
            for i in range(0, 2)
        }

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, _):
        star = random.randint(0, 1)
        idx = random.choice(self.rating_to_indexes[star])

        text = str(self.dataframe.loc[idx, "review_body"])
        label = self.dataframe.loc[idx, "star_rating"]

        x = tokenize(text).ids
        y = label

        x = torch.tensor(x, dtype=torch.long)

        return x, y


def generate_batch(data_batch, pad_idx):
    x_input, y_output = [], []
    for (x, y) in data_batch:
        x_input.append(x[:MAX_LEN])
        y_output.append(y)
    x_input = pad_sequence(x_input, padding_value=pad_idx, batch_first=True)
    y_output = torch.tensor(y_output, dtype=torch.long)

    return x_input, y_output


if __name__ == "__main__":
    batch_size = 1
    epochs = 10

    df = pd.DataFrame(
                  [
                  (0,"One Star"),
                  (0,"That was not great"),
                  (1,"Five Stars Phenomenal!"),
                  (1,"I loved that movie!"),
                  (0,"The story sucks hard"),
                  (0,"It was not entertaining"),
                  (1,"What an absolute masterpiece of modern art and modern cinematics!"),
                  (1,"Really good, my family and me liked it"),
                  (0,"Bad bad bad"),
                  (0,"That was not ok"),
                  (1,"I give five stars"),
                  (1,"I loved that movie!"),
                  (0,"The story sucks hard"),
                  (0,"It was not entertaining"),
                  (1,"Dude what an awesome perfomance")
                  ],
                  columns=["star_rating","review_body"]
                  )

    print(df.shape)
    print(df[["star_rating", "review_body"]])

    train_val, test = train_test_split(df, random_state=1337, test_size=0.5)
    train, val = train_test_split(train_val, random_state=1337, test_size=0.5)

    train_data = Dataset(dataframe=train)
    val_data = Dataset(dataframe=val)
    test_data = Dataset(dataframe=test)

    print("len(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=True,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=5,
        shuffle=False,
        collate_fn=partial(generate_batch, pad_idx=PAD_IDX),
    )

    base_path = Path(__file__).parents[1]

    model = Transformer(lr=1e-4, n_outputs=3, vocab_size=VOCAB_SIZE)

    # model.load_state_dict(torch.load(model_path)["state_dict"])

    # logger = TensorBoardLogger(
    #     save_dir=str(base_path / "logs"),
    #     name="sentiment",
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=DataDirHelper().getDataDir()+ 'companyTweets\\model',
        filename="sentiment",
        save_weights_only=True,
    )

    early_stopping = EarlyStopping(monitor="valid_loss", mode="min", patience=10)

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=False,
        callbacks=[checkpoint_callback, early_stopping],
        accumulate_grad_batches=1,
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, dataloaders=test_loader)
    
    #tokens, attr = attribution_fun("One fucking star", model,torch.device("cpu"))
    
    #print(attribution_to_html(tokens, attr))

    # DATALOADER: 0
    # TEST
    # RESULTS
    # {'test_loss': xxxx}
