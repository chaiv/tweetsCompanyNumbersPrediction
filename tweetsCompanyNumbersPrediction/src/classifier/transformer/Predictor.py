'''
Created on 08.02.2023

@author: vital
'''
from classifier.transformer.models import Transformer
from trainTransformerModel import Dataset, generate_batch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
import torch
from functools import partial
from captum.attr import LayerIntegratedGradients


class Predictor(object):

    def __init__(self, model: Transformer, tokenizer,deviceToUse = torch.device("cuda:0")):
        self.model = model
        self.tokenizer = tokenizer
        self.deviceToUse = deviceToUse
    
    def test(self, dataframe, batch_size = 1024, num_workers=16):
        test_data = Dataset(dataframe=dataframe,tokenizer = self.tokenizer)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=num_workers,
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
        
    def predictOne(self,sentence):
        x_Tokenids = self.tokenizer.encode(sentence)
        x = torch.tensor([x_Tokenids], dtype=torch.long).to(self.deviceToUse)
        with torch.no_grad():
            predicted = self.model(x)[0].argmax(0).item()
        return predicted
    
    def predictMultiple(self, sentences):
        x_Tokenids = [self.tokenizer.encode(sentence) for sentence in sentences]
        x_Tokenids = [torch.tensor(x, dtype=torch.long).to(self.deviceToUse) for x in x_Tokenids]
        x = torch.nn.utils.rnn.pad_sequence(x_Tokenids, batch_first=True, padding_value=0)
        with torch.no_grad():
            predicted = self.model(x).argmax(dim=1)
        return predicted
    
    def predictMultipleInChunks(self,sentences, chunkSize):
        predictions = []
        for i in range(0, len(sentences), chunkSize):
            chunk = sentences[i:i + chunkSize]
            chunkPredictions = self.predictMultiple(chunk)
            predictions += chunkPredictions.tolist()
            print(len(predictions))
        return predictions
    
    
    
    

    def calculateWordScores(self,text: str,observed_class):
        tokens = self.tokenizer.tokenize(text)
        tokens_idx = self.tokenizer.encode(text)
        tokens_idx += [self.tokenizer.getPADTokenID()]*(256 - len(tokens_idx))
        x = torch.tensor([tokens_idx], dtype=torch.long)
        ref = torch.tensor(
        [[self.tokenizer.getPADTokenID()] * (len(tokens_idx))], dtype=torch.long
        )
        lig = LayerIntegratedGradients(
            self.model,
            self.model.embeddings.embedding,
        )
        attributions_ig, delta = lig.attribute(
            x, ref, n_steps=500, return_convergence_delta=True, target=observed_class
        )
        attributions_ig = attributions_ig[0, 1:-1, :].sum(dim=-1).cpu()
        attributions_ig = attributions_ig / attributions_ig.abs().max()
        return tokens, attributions_ig.tolist()      
        
            