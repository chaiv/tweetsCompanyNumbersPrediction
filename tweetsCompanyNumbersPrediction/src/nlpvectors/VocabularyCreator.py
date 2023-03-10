'''
Created on 10.03.2023

@author: vital
'''
from collections import Counter
from nlpvectors.AbstractTokenizer import AbstractTokenizer

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class VocabularyCreator(object):

    def __init__(self, tokenizer : AbstractTokenizer):
        self.tokenizer = tokenizer

    def createVocabularyDict(self, sentences):
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        tokens = [token for sentence in tokenized_sentences for token in sentence]
        token_counts = Counter(tokens)
        vocab = {token: i+2 for i, (token, _) in enumerate(token_counts.most_common())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab
    
        