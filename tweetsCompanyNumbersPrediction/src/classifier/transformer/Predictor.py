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
from featureinterpretation.WordScoresWrapper import WordScoresWrapper



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
        
    def predictMultipleAsTweetGroups(self,sentenceWrappers):
        x = [torch.tensor(sentenceWrapper.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for sentenceWrapper in sentenceWrappers]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        with torch.no_grad():
            y_hat = self.model(x)
            _, predicted = torch.max(y_hat, 1)
            predicted_classes = predicted.tolist()
        return [self.predictionClassMapper.index_to_class(pred) for pred in predicted_classes]
    
    def predictMultipleAsTweetGroupsInChunks(self,sentenceWrappers, chunkSize):
        predictions = []
        for i in range(0, len(sentenceWrappers), chunkSize):
            chunk = sentenceWrappers[i:i + chunkSize]
            chunkPredictions = self.predictMultipleAsTweetGroups(chunk)
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
    
    
    def split_list_on_indices(self,lst, indices):
        if not indices:
            return lst
        
        splitted_list = []
        start_idx = 0
        for idx in indices:
            sublist = lst[start_idx:idx]
            if sublist:
                splitted_list.append(sublist)
            start_idx = idx + 1
        sublist = lst[start_idx:]
        if sublist:
            splitted_list.append(sublist)
    
        return splitted_list
    
    def calculateWordScoresOfTweetGroup(self,attributionsForSentenceWrapper,sentenceWrapper):
        separator_indexes = sentenceWrapper.getSeparatorIndexesInFeatureVector()
        attributionsForSentenceWrapperSplitted = self.split_list_on_indices(attributionsForSentenceWrapper.tolist(),separator_indexes)
        attribution_lists_for_sentences_of_wrapper = []
        for attribution_index in range(len(attributionsForSentenceWrapperSplitted)):
            numTokensOfSentence = len(sentenceWrapper.getTokens()[attribution_index])
            attributionsForSentence = attributionsForSentenceWrapperSplitted[attribution_index][:numTokensOfSentence]
            attribution_lists_for_sentences_of_wrapper.append(attributionsForSentence)
        return WordScoresWrapper(sentenceWrapper, attribution_lists_for_sentences_of_wrapper)
    
    def calculateWordScoresForTweetGroup(self, sentenceWrappers : list, observed_class,n_steps=500,internal_batch_size = 10):
        x = [torch.tensor(sentenceWrapper.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for sentenceWrapper in sentenceWrappers]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        ref_tokens = [[self.textEncoder.getPADTokenID()] * len(x[0])] * len(sentenceWrappers)
        ref = torch.tensor(ref_tokens, dtype=torch.long).to(self.deviceToUse)   
        attributionsOfAllSentenceWrappers = self.attributionsCalculator.attribute(x, ref, n_steps, observed_class, internal_batch_size)
        
        wordScoresWrappers = []
        
        for i in range(len(sentenceWrappers)):
            sentenceWrapper = sentenceWrappers[i]
            wordScoresWrappers.append(self.calculateWordScoresOfTweetGroup(attributionsOfAllSentenceWrappers[i],sentenceWrapper))
        
        return wordScoresWrappers
        
    
    
    
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
        
            