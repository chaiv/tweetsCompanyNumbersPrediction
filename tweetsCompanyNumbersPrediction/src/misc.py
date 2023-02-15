'''
Created on 05.02.2023

@author: vital
'''
import torch
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMapper import BinaryClassMapper

tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predictor = Predictor(model,tokenizer,BinaryClassMapper())
sentence1= "The free delivery gambit in retail  $AMZN $WMT $TGT $BBY"
sentence2 = "$AMZN News Updated Saturday, January 3, 2015 8:10:29 PM $NHMD $RPG $DXD $BIIB"
observed_class = 0
tokens,attributions = predictor.calculateWordScoresOne(sentence1, observed_class)
print(sum(attributions))
print(predictor.calculateWordScoresOneAsDict(sentence1, observed_class))
print(predictor.calculateWordScoresMultiple([sentence1,sentence2], observed_class))
#print(predictor.calculateWordScoresOne(sentence1, 1));

