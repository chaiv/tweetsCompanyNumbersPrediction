
import pandas as pd
from tweetpreprocess.DataDirHelper import DataDirHelper

df = pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\CompanyTweetsTeslaWithCarSales.csv')
print(len(df[df['class'] == 1.0]))
