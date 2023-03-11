'''
Created on 11.03.2023

@author: vital
'''
import math
import torch
import torch.nn as nn
class VocabIDTokenEmbedding(nn.Module):
    
    
    def __init__(self, vocab_size: int, emb_size):
        super(VocabIDTokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

        