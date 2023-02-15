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
from classifier.PredictionClassMapper import PredictionClassMapper


class Predictor(object):

    def __init__(self, model: Transformer, tokenizer,predictionClassMapper: PredictionClassMapper,deviceToUse = torch.device("cuda:0") ):
        self.model = model
        self.tokenizer = tokenizer
        self.deviceToUse = deviceToUse
        self.predictionClassMapper = predictionClassMapper
    
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
        
    def predictOne(self, sentence):
        x_Tokenids = self.tokenizer.encode(sentence)
        x = torch.tensor([x_Tokenids], dtype=torch.long).to(self.deviceToUse)
        with torch.no_grad():
            y_hat = self.model(x)
            _,predicted = torch.max(y_hat, 1)
            predicted_class = predicted.item()
        return self.predictionClassMapper.index_to_class(predicted_class)
    
    
    def predictMultiple(self, sentences):
        x_Tokenids = [self.tokenizer.encode(sentence) for sentence in sentences]
        x_Tokenids = [torch.tensor(x, dtype=torch.long).to(self.deviceToUse) for x in x_Tokenids]
        x = torch.nn.utils.rnn.pad_sequence(x_Tokenids, batch_first=True, padding_value=0)
        with torch.no_grad():
            y_hat = self.model(x)
            _, predicted = torch.max(y_hat, 1)
            predicted_classes = predicted.tolist()
            return [self.predictionClassMapper.index_to_class(pred) for pred in predicted_classes]
    
    def predictMultipleInChunks(self,sentences, chunkSize):
        predictions = []
        for i in range(0, len(sentences), chunkSize):
            chunk = sentences[i:i + chunkSize]
            chunkPredictions = self.predictMultiple(chunk)
            predictions += chunkPredictions.tolist()
            print(len(predictions))
        return predictions
    
    
    def calculateWordScoresOneAsDict(self,sentence: str,observed_class):
        tokens, attributions = self.calculateWordScoresOne(sentence,observed_class)
        token_to_attrib = {}
        for token, attribution in zip(tokens, attributions):
            token_to_attrib[token] = attribution
        return token_to_attrib
    

    def calculateWordScoresOne(self,sentence: str,observed_class):
        tokens = self.tokenizer.tokenize(sentence)
        tokens_idx = self.tokenizer.encode(sentence)
        tokens_idx += [self.tokenizer.getPADTokenID()]*(256 - len(tokens_idx))
        x = torch.tensor([tokens_idx], dtype=torch.long).to(self.deviceToUse)
        ref = torch.tensor(
        [[self.tokenizer.getPADTokenID()] * (len(tokens_idx))], dtype=torch.long
        ).to(self.deviceToUse)
        lig = LayerIntegratedGradients(
            self.model,
            self.model.embeddings.embedding,
        )
        attributions_ig, delta = lig.attribute(
            x, ref, n_steps=500, return_convergence_delta=True, target=observed_class
        )
        attributions_ig = attributions_ig[0, :, :].sum(dim=-1).cpu()
        attributions_ig = attributions_ig / attributions_ig.abs().max()
        return tokens, attributions_ig.tolist()      
    
    def calculateWordScoresMultiple(self, sentences, observed_class):
        token_lists = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        tokens_idxs = [self.tokenizer.encode(sentence) for sentence in sentences]
        padded_tokens_idxs = []
        for tokens_idx in tokens_idxs:
            tokens_idx += [self.tokenizer.getPADTokenID()] * (256 - len(tokens_idx))
            padded_tokens_idxs.append(tokens_idx)
        x = torch.tensor(padded_tokens_idxs, dtype=torch.long).to(self.deviceToUse)
        ref_tokens = [[self.tokenizer.getPADTokenID()] * 256] * len(sentences)
        ref = torch.tensor(
            ref_tokens, dtype=torch.long
            ).to(self.deviceToUse)            
        lig = LayerIntegratedGradients(
            self.model,
            self.model.embeddings.embedding,
            )
        attributions_ig, delta = lig.attribute(
            x, ref, n_steps=500, return_convergence_delta=True, target=observed_class
            )
        attributions_ig = attributions_ig[:, :, :].sum(dim=-1).cpu()
        attributions_ig = attributions_ig / attributions_ig.abs().max(dim=1, keepdim=True)[0]
        scores = []
        for i in range(len(sentences)):
            tokens = token_lists[i]
            attributions_ig_list = attributions_ig[i].tolist()
            scores.append((tokens, attributions_ig_list))
        return scores
        
            