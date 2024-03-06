'''
Created on 06.03.2024

@author: vital
'''
import torch
import pytorch_lightning as pl

class LSTMNNWithDropout(pl.LightningModule):
    def __init__(self, emb_size, word_vectors):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor(word_vectors.vectors), freeze=False)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.2)  # Add dropout to LSTM
        self.fc1 = torch.nn.Linear(512, 512)
        self.dropout1 = torch.nn.Dropout(0.5)  # Add dropout layer
        self.fc2 = torch.nn.Linear(512, 256)
        self.dropout2 = torch.nn.Dropout(0.5)  # Add dropout layer
        self.fc3 = torch.nn.Linear(256, 2)
        self.batchnorm1 = torch.nn.BatchNorm1d(512)  # Add batch normalization
        self.batchnorm2 = torch.nn.BatchNorm1d(256)  # Add batch normalization
        
    def forward(self, inputs):
        x = self.embedding(inputs)
        _, (h_n, _) = self.lstm(x)
        x = self.batchnorm1(self.fc1(h_n[-1]))  # Add batch normalization
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)  # Add dropout
        x = self.batchnorm2(self.fc2(x))  # Add batch normalization
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)  # Add dropout
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Dynamic LR adjustment
        return [optimizer], [lr_scheduler]