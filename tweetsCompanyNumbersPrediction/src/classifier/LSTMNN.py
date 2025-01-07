'''
Created on 12.03.2023

@author: vital
'''
import torch
import pytorch_lightning as pl

class LSTMNN(pl.LightningModule):
    def __init__(self, emb_size, word_vectors,num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(word_vectors.vectors), freeze=False)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size=512, num_layers=2, batch_first=True)
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)
        
    def forward(self, inputs):
        x = self.embedding(inputs)
        _, (h_n, _) = self.lstm(x)
        x = self.fc1(h_n[-1])
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
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