'''
Created on 12.03.2023

@author: vital
'''
import torch
import pytorch_lightning as pl
class FeedForwardNN(pl.LightningModule):
    def __init__(self, emb_size,word_vectors):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(word_vectors.vectors), freeze=False)
        self.fc1 = torch.nn.Linear(emb_size, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        
    def forward(self, inputs):
        x = self.embedding(inputs)
        x = x.mean(dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    