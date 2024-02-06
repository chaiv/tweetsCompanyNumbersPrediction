
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.DataframeSplitter import DataframeSplitter
from sklearn.model_selection._split import KFold


df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsTeslaWithCarSales.csv')
df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
print(TweetDataframeExplore(df).getClassDistribution())
splitter = DataframeSplitter()
tweetSplits = splitter.getSplitIds(df, 5) #how many tweets should be trained as one sample
kfold_splits = 3
kfold_cross_val = KFold(n_splits=kfold_splits, shuffle=True, random_state=1337)
for fold, (train_idx, test_idx) in enumerate(kfold_cross_val.split(tweetSplits)):
    oneClassCount = 0
    zeroClassCount = 0
    for idx in test_idx:
        split = tweetSplits[idx]
        splitDf =  df[df["tweet_id"].isin( split)]
        if(splitDf["class"].iloc[0]==0.0):
            zeroClassCount=zeroClassCount+1
        else: 
            oneClassCount =oneClassCount+1
    print("zero: ",zeroClassCount," one: ",oneClassCount)



