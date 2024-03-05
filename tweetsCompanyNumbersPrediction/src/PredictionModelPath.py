'''
Created on 23.02.2024

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper


class PredictionModelPath(object):

    def __init__(self, 
                 nasdaqTag, 
                 tweetDataframePath, 
                 financialNumbersPath, 
                 wordvectorsPath, 
                 predictionModelPath,
                 top2vecModelPath,
                 tweetGroupSize
                 ):
        self.nasdaqTag = nasdaqTag
        self.dataframePath = tweetDataframePath
        self.financialNumbersPath = financialNumbersPath
        self.wordvectorsPath = wordvectorsPath
        self.modelPath = predictionModelPath
        self.top2vecModelPath =  top2vecModelPath
        self.tweetGroupSize = tweetGroupSize
        
    def getNasdaqTag(self):
        return self.nasdaqTag
        
    def getDataframePath(self):
        return self.dataframePath
    
    def getFinancialNumbersPath(self):
        return self.financialNumbersPath
    
    def getWordVectorsPath(self):
        return self.wordvectorsPath
    
    def getModelPath(self):
        return  self.modelPath
    
    def getTweetGroupSize(self):
        return self.tweetGroupSize
    
    def getTop2VecModelPath(self):
        return self.top2vecModelPath
    
MICROSOFT_EPS_5 = PredictionModelPath(
    "MSFT",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    5
    ) 

MICROSOFT_EPS_10 = PredictionModelPath(
      "MSFT",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    10
    ) 

MICROSOFT_GROSS_PROFIT_20 = PredictionModelPath(
      "MSFT",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftGrossProfit.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftGrossProfitLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    20
    ) 

MICROSOFT_XBOX_USERS_20 = PredictionModelPath(
      "MSFT",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithXboxUsers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftXboxUsers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftXboxUsersLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    20
    ) 
    
APPLE_IPHONE_SALES_5 = PredictionModelPath(
    "AAPL",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    5
    )

AMAZON_REVENUE_5 = PredictionModelPath(
    "AMZN",
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN5',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    5
    )

TESLA_CAR_SALES_5 = PredictionModelPath(
    "TSLA",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    5
    )

APPLE_IPHONE_SALES_10 = PredictionModelPath(
    "AAPL",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    10
    )

AMAZON_REVENUE_10 = PredictionModelPath(
    "AMZN",
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    10
    )

TESLA_CAR_SALES_10 = PredictionModelPath(
    "TSLA",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    10
    )



APPLE__IPHONE_SALES_20 = PredictionModelPath(
    "AAPL",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    20
    )

AMAZON_REVENUE_20 = PredictionModelPath(
    "AMZN",
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    20
    )

TESLA_CAR_SALES_20 = PredictionModelPath(
    "TSLA",
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    20
    )

        