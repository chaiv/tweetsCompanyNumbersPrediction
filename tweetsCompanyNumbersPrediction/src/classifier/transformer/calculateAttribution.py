'''
Created on 06.02.2023

@author: vital
'''
import torch
from classifier.transformer.models import Transformer
from classifier.transformer.nlp_utils import VOCAB_SIZE
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.transformer.predict_utils import attribution_fun,attribution_to_html
model = Transformer(lr=1e-4, n_outputs=3, vocab_size= VOCAB_SIZE)
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\sentiment.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
predicted, tokens, attr = attribution_fun("One fucking star", model,torch.device("cpu"))
print(predicted, tokens, attr)
print(attribution_to_html(tokens, attr))