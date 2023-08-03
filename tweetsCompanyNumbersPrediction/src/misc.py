
import spacy
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
nlp = spacy.load('en_core_web_sm')
df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweets.csv')
for doc in nlp.pipe(df["body"]):
    print(doc.ents)


  