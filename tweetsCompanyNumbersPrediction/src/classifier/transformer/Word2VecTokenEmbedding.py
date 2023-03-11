'''
Created on 11.03.2023

@author: vital
'''
import math
import torch
import torch.nn as nn
class Word2VecTokenEmbedding(nn.Module):
    
    
    def __init__(self, word_vectors, emb_size,pad_token_id):
        self.word_vectors = word_vectors
        super(Word2VecTokenEmbedding, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding.from_pretrained(embeddings = word_vectors,padding_idx=pad_token_id)

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        