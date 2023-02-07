'''
Created on 05.02.2023

@author: vital
'''
import torch
from classifier.transformer.models import Transformer
from nlpvectors.TokenizerTop2Vec import TokenizerTop2Vec
from tweetpreprocess.DataDirHelper import DataDirHelper
from captum.attr import LayerIntegratedGradients

def attribution_fun(tokenizer,base_class, text: str, model: Transformer):
    tokens = tokenizer.tokenize(text)
    tokens_idx = tokenizer.encode(text)
    tokens_idx += [tokenizer.getPADTokenID()]*(256 - len(tokens_idx))
    x = torch.tensor([tokens_idx], dtype=torch.long)
    ref = torch.tensor(
        [[tokenizer.getPADTokenID()] * (len(tokens_idx))], dtype=torch.long
    )
    len_x = len(x)
    len_ref = len(ref)
    
    lig = LayerIntegratedGradients(
        model,
        model.embeddings.embedding,
    )

    attributions_ig, delta = lig.attribute(
        x, ref, n_steps=500, return_convergence_delta=True, target=base_class
    )
    attributions_ig = attributions_ig[0, 1:-1, :].sum(dim=-1).cpu()
    attributions_ig = attributions_ig / attributions_ig.abs().max()
    return tokens[1:-1], attributions_ig.tolist()


tokenizer = TokenizerTop2Vec(DataDirHelper().getDataDir()+ "companyTweets\TokenizerAmazon.json")
vocab_size = tokenizer.getVocabularyLength()
model = Transformer(lr=1e-4, n_outputs=2, vocab_size=vocab_size+2)
checkpoint = torch.load(DataDirHelper().getDataDir()+"companyTweets\\model\\amazonTweetpredict39Epochs.ckpt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
text = "$AMZN - 21st Century Fox Earnings: An EPS Beat, but Revenue Misses Expectations" 
#text = "S&P100 #Stocks Performance $HD $LOW $SBUX $TGT $DVN $IBM $AMZN $F $APA $GM $MS $HAL $DIS $MCD $BMY $XOM  more@"
#print(attribution_fun(tokenizer,1,text, model))
x_Tokenids = tokenizer.encode(text)
x = torch.tensor([x_Tokenids], dtype=torch.long)
with torch.no_grad():
    predicted = model(x)[0].argmax(0).item()
    print(predicted)
