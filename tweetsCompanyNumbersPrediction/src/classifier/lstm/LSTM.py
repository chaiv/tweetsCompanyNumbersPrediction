'''
Created on 19.01.2023

@author: vital
'''
import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.ModuleList):
    
    
    def create_emb_layer(self,weights_matrix, non_trainable=False):
        num_embeddings = np.size(weights_matrix,0)
        embedding_dim = np.size(weights_matrix,1)
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': 
                                   torch.from_numpy(weights_matrix)
                                   })
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim
    
    
    

    def __init__(self,weights_matrix, batch_size=64, hidden_dim =128,lstm_layers=2,input_size=300 ):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = input_size
        
        self.dropout = nn.Dropout(0.5)
        self.embedding, num_embeddings, embedding_dim = self.create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim*2)
        self.fc2 = nn.Linear(self.hidden_dim*2, 1)
        
    def forward(self, x):
    
        h = torch.zeros((self.LSTM_layers, self.input_size, self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, self.input_size, self.hidden_dim))
        
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(x)

        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)

        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out
        