'''
Created on 05.02.2023

@author: vital
'''
import torch
import pandas as pd
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.Predictor import Predictor
from classifier.PredictionClassMappers import BINARY_0_1
from featureinterpretation.TokenAttributionStore import ImportantWordStore
from featureinterpretation.AttributionsCalculator import AttributionsCalculator
tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
model = model.to(torch.device("cuda:0"))
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetPredict.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()  
predictionClassMapper = BINARY_0_1 
attributionCalculator = AttributionsCalculator(model,model.embeddings.embedding)
predictor = Predictor(model,tokenizer,predictionClassMapper,attributionCalculator)
df = pd.DataFrame(
                  [
                  ("id1","company next charger will automatically connect to your car"),
                  ("id2","Bank Of Dummy Just Released A New Auto Report And They Discussed companies: Bank of Dummy"),
                  ("id3","Electric Vehicle Battery Energy Density - Accelerating the Development Timeline."),
                  ("id4","Interview: World Needs Hundreds of Gigafactories."),
                  ("id5","Great meeting you soon")
                  ],
                  columns=["tweet_id","body"]
                  )

tokenAttributionStore = ImportantWordStore()
tweet_ids = df["tweet_id"].tolist()
sentences = df["body"].tolist()
word_scores = predictor.calculateWordScores(sentences, 1)
tweet_ids_with_word_scores = [(x,) + y for x, y in zip(tweet_ids, word_scores)]
tokenAttributionStore.add_multiple_data(tweet_ids_with_word_scores)    
print(tokenAttributionStore.to_dataframe())


