'''
Created on 10.03.2023

@author: vital
'''
from collections import Counter
from nlpvectors.AbstractTokenizer import AbstractTokenizer

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SEP_TOKEN = '<SEP>'


class VocabularyCreator(object):

    def __init__(self, tokenizer : AbstractTokenizer):
        self.tokenizer = tokenizer

    def createVocabularyDict(self, sentences):
        tokenized_sentences = [self.tokenizer.tokenize(sentence) for sentence in sentences]
        tokens = [token for sentence in tokenized_sentences for token in sentence]
        token_counts = Counter(tokens)
        vocab = {token: i for i, (token, _) in enumerate(token_counts.most_common())}
        specialTokens = [PAD_TOKEN,UNK_TOKEN,SEP_TOKEN]
        for token in specialTokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        return vocab
    
        