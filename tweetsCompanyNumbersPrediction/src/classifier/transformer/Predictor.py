'''
Created on 08.02.2023

@author: vital
'''
from classifier.transformer.models import Transformer
import torch
from classifier.PredictionClassMapper import PredictionClassMapper
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
from nlpvectors.AbstractEncoder import AbstractEncoder
from nlpvectors.AbstractTokenizer import AbstractTokenizer



class Predictor(object):

    def __init__(self, 
                 model: Transformer, 
                 tokenizer: AbstractTokenizer,
                 textEncoder : AbstractEncoder,
                 predictionClassMapper: PredictionClassMapper,
                 attributionsCalculator:  AttributionsCalculator,
                 deviceToUse = torch.device("cuda:0") 
                 ):
        self.model = model
        self.tokenizer = tokenizer
        self.textEncoder = textEncoder
        self.attributionsCalculator = attributionsCalculator
        self.deviceToUse = deviceToUse
        self.predictionClassMapper = predictionClassMapper
        
    def predictMultipleAsWrapper(self,sentenceWrappers):
        x = [torch.tensor(sentenceWrapper.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for sentenceWrapper in sentenceWrappers]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        with torch.no_grad():
            y_hat = self.model(x)
            _, predicted = torch.max(y_hat, 1)
            predicted_classes = predicted.tolist()
        return [self.predictionClassMapper.index_to_class(pred) for pred in predicted_classes]
    
    def predictMultipleAsWrappersInChunks(self,sentenceWrappers, chunkSize):
        predictions = []
        for i in range(0, len(sentenceWrappers), chunkSize):
            chunk = sentenceWrappers[i:i + chunkSize]
            chunkPredictions = self.predictMultipleAsWrapper(chunk)
            predictions += chunkPredictions
            print("Chunks processed",len(predictions))
        return predictions
    

    def predictMultiple(self, sentences):
        x_Tokenids = [self.textEncoder.encodeTokens(self.tokenizer.tokenize(sentence)) for sentence in sentences]
        x_Tokenids = [torch.tensor(x, dtype=torch.long).to(self.deviceToUse) for x in x_Tokenids]
        x = torch.nn.utils.rnn.pad_sequence(x_Tokenids, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        with torch.no_grad():
            y_hat = self.model(x)
            _, predicted = torch.max(y_hat, 1)
            predicted_classes = predicted.tolist()
        return [self.predictionClassMapper.index_to_class(pred) for pred in predicted_classes]
    
    def predictMultipleInChunks(self,sentences, chunkSize):
        predictions = []
        for i in range(0, len(sentences), chunkSize):
            chunk = sentences[i:i + chunkSize]
            chunkPredictions = self.predictMultiple(chunk)
            predictions += chunkPredictions
            print("Chunks processed",len(predictions))
        return predictions
    
    def calculateWordScoresInChunks(self, sentences: list, observed_class, chunk_size, n_steps=500, internal_batch_size=10):
        token_indexes_lists = []
        token_lists = []
        attributions_lists = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            chunk_token_indexes, chunk_token_lists, chunk_attributions = self.calculateWordScores(chunk, observed_class, n_steps, internal_batch_size)
            token_indexes_lists +=  chunk_token_indexes
            token_lists += chunk_token_lists
            attributions_lists += chunk_attributions
            print("Chunks processed",len(token_indexes_lists))
        return token_indexes_lists, token_lists, attributions_lists
    
    
    def calculateWordScoresForWrappers(self, sentenceWrappers : list, observed_class,n_steps=500,internal_batch_size = 10):
        x = [torch.tensor(sentenceWrapper.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for sentenceWrapper in sentenceWrappers]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        ref_tokens = [[self.textEncoder.getPADTokenID()] * len(x[0])] * len(sentenceWrappers)
        ref = torch.tensor(ref_tokens, dtype=torch.long).to(self.deviceToUse)   
        attributions_ig = self.attributionsCalculator.attribute(x, ref, n_steps, observed_class, internal_batch_size)
        
        total_token_indexes_lists = []
        total_token_lists = []
        total_attribution_lists = []
        
        for i in range(len(sentenceWrappers)):
            sentenceWrapper = sentenceWrappers[i]
            attributionsForSentenceWrapper = attributions_ig[i]
            attributionsForSentenceWrapperSplitted = #split  attributionsForSentenceWrapper on values of sentenceWrapper.getSeparatorIndexesInFeatureVector
            total_token_indexes_lists  = total_token_indexes_lists + sentenceWrapper.getTokenIndexesLists()
            total_token_lists = total_token_lists + sentenceWrapper.getTokenLists()
            for token_list in sentenceWrapper.getTokenLists():
                #num_tokens = len(sentenceWrapper.getTokenLists())
                #attributions for token
            total_attribution_lists =  #
            
        
        
        pass
        
    
    
    
    def calculateWordScores(self, sentences : list, observed_class,n_steps=500,internal_batch_size = 10):
        index_token_lists = [self.tokenizer.tokenizeWithIndex(sentence) for sentence in sentences]
        indexes_lists, token_lists = zip(*index_token_lists)
        tokens_idxs = [self.textEncoder.encodeTokens(tokens) for tokens in token_lists]
        padded_tokens_idxs = []
        for tokens_idx in tokens_idxs:
            tokens_idx += [self.textEncoder.getPADTokenID()] * (self.textEncoder.getMaxWordsAmount() - len(tokens_idx))
            padded_tokens_idxs.append(tokens_idx)
        x = torch.tensor(padded_tokens_idxs, dtype=torch.long).to(self.deviceToUse)
        ref_tokens = [[self.textEncoder.getPADTokenID()] * self.textEncoder.getMaxWordsAmount()] * len(sentences)
        ref = torch.tensor(
            ref_tokens, dtype=torch.long
            ).to(self.deviceToUse)            
        attributions_ig = self.attributionsCalculator.attribute(x, ref, n_steps, observed_class, internal_batch_size)
        attributions_ig_trimmed = []
        for i, attributions in enumerate(attributions_ig):
            num_tokens = len(token_lists[i])
            attributions_trimmed = attributions[:num_tokens]
            attributions_ig_trimmed.append(attributions_trimmed.tolist())
        return indexes_lists, token_lists, attributions_ig_trimmed
        
            