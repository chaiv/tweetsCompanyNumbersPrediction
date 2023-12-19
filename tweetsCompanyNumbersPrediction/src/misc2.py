import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper


tweets = pd.read_csv (DataDirHelper().getDataDir()+ "companyTweets\\CompanyTweetsAAPLFirst1000.csv")

index = tweets .index[tweets ['tweet_id'] ==550443857595600896]
print(index)
print(tweets.iloc[index]['tweet_id'])