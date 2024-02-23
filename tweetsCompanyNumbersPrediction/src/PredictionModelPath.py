'''
Created on 23.02.2024

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper


class PredictionModelPath(object):

    def __init__(self, 
                 dataframePath, 
                 wordvectorsPath, 
                 modelPath,
                 tweetGroupSize
                 ):
        self.dataframePath = dataframePath
        self.wordvectorsPath = wordvectorsPath
        self.modelPath = modelPath
        self.tweetGroupSize = tweetGroupSize
        
        
    def getDataframePath(self):
        return self.dataframePath
    
    def getWordVectorsPath(self):
        return self.wordvectorsPath
    
    def getModelPath(self):
        return  self.modelPath
    
    def getTweetGroupSize(self):
        return self.tweetGroupSize
    
APPLE_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM5",
    5
    )

AMAZON_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN5',
    5
    )

TESLA_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5',
    5
    )

APPLE_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM10",
    10
    )

AMAZON_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10',
    10
    )

TESLA_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10',
    10
    )



APPLE_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM20",
    20
    )

AMAZON_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20',
    20
    )

TESLA_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM20',
    20
    )

        