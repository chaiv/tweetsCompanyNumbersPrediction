'''
Created on 15.11.2023

@author: vital
'''
import spacy
from spacy.tokens import Doc



class POSTag(object):
    def __init__(self,token,posTag):
        self.token = token
        self.posTag = posTag
        
    def getToken(self):
        return self.token
    
    def getPosTag(self):
        return self.posTag
    

class PartOfSpeechTagging(object):

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm') 
        self.nlp.tokenizer = self.custom_tokenizer
    
    def custom_tokenizer(self,text):
    # Custom because otherwise spacy splits such tokens as $AMZN into two $ and AMZN
        tokens = text.split()
        return Doc(self.nlp.vocab, words=tokens)

    
    
    def getPOSTagsAsStrList(self,sentence)-> list[str]: 
        posTags = self.getPOSTags(sentence)
        return [posTag.getPosTag() for posTag in  posTags]

    def getPOSTags(self,sentence)-> list[POSTag]:
        doc = self.nlp(sentence)
        posTags = []
        for token in doc:  
            posTags.append(
                POSTag(token.text,token.pos_)
                )  
        return posTags