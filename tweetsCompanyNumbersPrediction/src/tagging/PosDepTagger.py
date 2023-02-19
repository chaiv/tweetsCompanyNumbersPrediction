'''
Created on 19.02.2023

@author: vital
'''

import spacy

class PosDepTagger(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def get_tags(self, sentence):
        doc = self.nlp(sentence)
        tags = []
        for i, token in enumerate(doc):
            tags.append((i, token.text, token.pos_, token.dep_))
        return tags