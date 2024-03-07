'''
Created on 23.02.2024

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper


class PredictionModelPath(object):

    def __init__(self, 
                 nasdaqTag : list[str], 
                 tweetDataframePath : str, 
                 financialNumbersPath : str, 
                 wordvectorsPath : str, 
                 predictionModelPath : str,
                 top2vecModelPath : str,
                 isEqualSamplesForEachClass : bool, 
                 tweetGroupSize: int
                 ):
        self.nasdaqTag = nasdaqTag
        self.dataframePath = tweetDataframePath
        self.financialNumbersPath = financialNumbersPath
        self.wordvectorsPath = wordvectorsPath
        self.modelPath = predictionModelPath
        self.top2vecModelPath =  top2vecModelPath
        self.isEqualSamplesForEachClass = isEqualSamplesForEachClass
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
    
    def hasEqualSamplesForEachClass(self):
        return self.isEqualSamplesForEachClass 
    
MICROSOFT_EPS_5 = PredictionModelPath(
    ["MSFT"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    True,
    5
    ) 

MICROSOFT_EPS_10 = PredictionModelPath(
    ["MSFT"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftEpsLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    True,
    10
    ) 

MICROSOFT_GROSS_PROFIT_20 = PredictionModelPath(
    ["MSFT"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftGrossProfit.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftGrossProfitLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    True,
    20
    ) 


MICROSOFT_XBOX_USERS_10 = PredictionModelPath(
    ["MSFT"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithXboxUsers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftXboxUsers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftXboxUsersLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    True,
    10
    ) 

MICROSOFT_XBOX_USERS_20 = PredictionModelPath(
    ["MSFT"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsMicrosoftWithXboxUsers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\microsoftXboxUsers.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsMicrosoft.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\microsoftXboxUsersLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\microsoftTopicModel",
    True,
    20
    ) 
    
APPLE_IPHONE_SALES_5 = PredictionModelPath(
    ["AAPL"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM5",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    True,
    5
    )

AMAZON_REVENUE_5 = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN5',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    False,
    5
    )

TESLA_CAR_SALES_5 = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    5
    )

APPLE_IPHONE_SALES_10 = PredictionModelPath(
    ["AAPL"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    True,
    10
    )

AMAZON_REVENUE_10 = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    10
    )

TESLA_CAR_SALES_10 = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    10
    )



APPLE__IPHONE_SALES_20 = PredictionModelPath(
    ["AAPL"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithIPhoneSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleIphoneSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleIphoneSalesLSTM20",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    True,
    20
    )

AMAZON_REVENUE_20 = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    20
    )

TESLA_CAR_SALES_20 = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    20
    )

GOOGLE_SE_MARKET_SHARE_5 = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    5
    )

GOOGLE_SE_MARKET_SHARE_10 = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    10
    )

GOOGLE_SE_MARKET_SHARE_20 = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    20
    )


        