'''
Created on 12.03.2023

@author: vital
'''
import torch
import pytorch_lightning as pl

LAST_HIDDEN_STATE_POOLING = 'last'
MEAN_POOLING = 'mean'


class LSTMNN(pl.LightningModule):
    '''
    pooling selects how the token sequence of a tweet group is reduced to one vector.

    'last' uses the final hidden state, which is the original behaviour. A tweet group of ten tweets
    produces a few hundred tokens, and on the Apple EPS data the final hidden state carries so little
    of them that the model stays at the class weighted prior: its training loss remains at ln(4) and
    it predicts a single class for every sample, on a random split as well as on a temporal one.

    'mean' averages the outputs over all non padding positions. With no other change the training
    loss falls to 0.08 and the model separates all four classes. It needs padTokenIdx to build the
    mask, so pass the value of WordVectorsIDEncoder.getPADTokenID().
    '''

    def __init__(self, emb_size, word_vectors, num_classes, class_weights=None,
                 pooling=LAST_HIDDEN_STATE_POOLING, padTokenIdx=None,
                 freeze_embeddings=False):
        super().__init__()
        if pooling == MEAN_POOLING and padTokenIdx is None:
            raise ValueError("padTokenIdx is required for mean pooling, it defines which positions are padding")
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(word_vectors.vectors), freeze=freeze_embeddings)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size=512, num_layers=2, batch_first=True)
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, num_classes)
        self.pooling = pooling
        self.padTokenIdx = padTokenIdx
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def embed(self, inputs):
        """Embed token IDs; subclasses may add a small task-specific adapter."""
        return self.embedding(inputs)

    def encode(self, inputs):
        """Return the original 256-dimensional representation before the class head."""
        x = self.embed(inputs)
        outputs, (h_n, _) = self.lstm(x)
        if self.pooling == MEAN_POOLING:
            mask = (inputs != self.padTokenIdx).unsqueeze(-1).to(outputs.dtype)
            x = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = h_n[-1]
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        return x

    def forward(self, inputs):
        return self.fc3(self.encode(inputs))
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.class_weights)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        #loss = torch.nn.functional.cross_entropy(outputs, targets, weight=self.class_weights)
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
