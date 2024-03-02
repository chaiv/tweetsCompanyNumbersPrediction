'''
Created on 23.02.2024

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper


class PredictionModelPath(object):

    def __init__(self, 
                 dataframePath, 
                 wordvectorsPath, 
                 predictionModelPath,
                 top2vecModelPath,
                 tweetGroupSize
                 ):
        self.dataframePath = dataframePath
        self.wordvectorsPath = wordvectorsPath
        self.modelPath = predictionModelPath
        self.top2vecModelPath =  top2vecModelPath
        self.tweetGroupSize = tweetGroupSize
        
        
    def getDataframePath(self):
        return self.dataframePath
    
    def getWordVectorsPath(self):
        return self.wordvectorsPath
    
    def getModelPath(self):
        return  self.modelPath
    
    def getTweetGroupSize(self):
        return self.tweetGroupSize
    
    def getTop2VecModelPath(self):
        return self.top2vecModelPath
    
MICROSOFT_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    5
    ) 

MICROSOFT_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    10
    ) 
    
APPLE_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    5
    )

AMAZON_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN5',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    5
    )

TESLA_5 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    5
    )

APPLE_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    10
    )

AMAZON_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    10
    )

TESLA_10 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    10
    )



APPLE_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    20
    )

AMAZON_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    20
    )

TESLA_20 = PredictionModelPath(
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    20
    )

        