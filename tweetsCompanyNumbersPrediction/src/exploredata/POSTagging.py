'''
Created on 15.11.2023

@author: vital
'''
import spacy

class PartOfSpeechTagging(object):



    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm') 

    def getPOSTags(self,sentence)-> list[str]:
        doc = self.nlp(sentence)
        posTags = []
        for token in doc:  
            posTags.append(token.pos_)  
        return posTags