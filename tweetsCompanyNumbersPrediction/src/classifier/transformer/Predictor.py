'''
Created on 08.02.2023

@author: vital
'''
from classifier.transformer.models import Transformer
from trainTransformerModel import Dataset, generate_batch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from functools import partial

class Predictor(object):

    def __init__(self, model: Transformer, tokenizer, batch_size = 1024, num_workers=16):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def test(self, dataframe):
        test_data = Dataset(dataframe=dataframe,tokenizer = self.tokenizer)
        test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=partial(generate_batch, pad_idx=self.tokenizer.getPADTokenID()),
        )
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator='gpu',
            devices=1,
            logger=False,
            accumulate_grad_batches=1,
        )
        trainer.test(self.model, dataloaders=test_loader)    
        
            