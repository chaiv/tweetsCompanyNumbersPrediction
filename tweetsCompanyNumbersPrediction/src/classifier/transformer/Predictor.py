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
from torch.ao.quantization import observer



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
        
    def predictMultipleAsTweetGroups(self,tweetGroups):
        x = [torch.tensor(tweetGroup.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for tweetGroup in tweetGroups]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        with torch.no_grad():
            y_hat = self.model(x)
            _, predicted = torch.max(y_hat, 1)
            predicted_classes = predicted.tolist()
        return [self.predictionClassMapper.index_to_class(pred) for pred in predicted_classes]
    
    def predictMultipleAsTweetGroupsInChunks(self,tweetGroups, chunkSize):
        predictions = []
        for i in range(0, len(tweetGroups), chunkSize):
            chunk = tweetGroups[i:i + chunkSize]
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
    
    def calculateWordScoresOfTweetGroup(self,attributionsForTweetGroup,tweetGroup):
        separator_indexes = tweetGroup.getSeparatorIndexesInFeatureVector()
        attributionsForTweetGroupSplitted = self.split_list_on_indices(attributionsForTweetGroup.tolist(),separator_indexes)
        attribution_lists_for_sentences_of_wrapper = []
        for attribution_index in range(len(attributionsForTweetGroupSplitted)):
            numTokensOfSentence = len(tweetGroup.getTokens()[attribution_index])
            attributionsForSentence = attributionsForTweetGroupSplitted[attribution_index][:numTokensOfSentence]
            attribution_lists_for_sentences_of_wrapper.append(attributionsForSentence)
        return WordScoresWrapper(tweetGroup, attribution_lists_for_sentences_of_wrapper)
    
    def calculateWordScoresOfTweetGroups(self, tweetGroups : list, observed_class,n_steps=500,internal_batch_size = 10):
        x = [torch.tensor(tweetGroup.getFeatureVector(), dtype=torch.long).to(self.deviceToUse) for tweetGroup in tweetGroups]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.textEncoder.getPADTokenID())
        ref_tokens = [[self.textEncoder.getPADTokenID()] * len(x[0])] * len(tweetGroups)
        ref = torch.tensor(ref_tokens, dtype=torch.long).to(self.deviceToUse)   
        attributionsOfAllTweetGroups = self.attributionsCalculator.attribute(x, ref, n_steps, observed_class, internal_batch_size)
        wordScoresWrappers = []
        for i in range(len(tweetGroups)):
            tweetGroup = tweetGroups[i]
            wordScoresWrappers.append(self.calculateWordScoresOfTweetGroup(attributionsOfAllTweetGroups[i],tweetGroup))
        return wordScoresWrappers
    
    
    def calculateWordScoresOfTweetGroupsInChunks(self, tweetGroups : list, observed_class,chunkSize, n_steps=500,internal_batch_size = 10):
        wordScoresWrappers = []
        for i in range(0, len(tweetGroups), chunkSize):
            chunk = tweetGroups[i:i + chunkSize]
            chunkWordScoresWrappers = self.calculateWordScoresOfTweetGroups(chunk,observed_class,n_steps,internal_batch_size)
            wordScoresWrappers += chunkWordScoresWrappers
            print("Chunks processed",len(chunkWordScoresWrappers))
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
        
            