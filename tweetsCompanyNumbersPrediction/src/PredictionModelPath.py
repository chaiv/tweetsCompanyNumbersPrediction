'''
Created on 23.02.2024

@author: vital
'''
from tweetpreprocess.DataDirHelper import DataDirHelper
from classifier.PredictionClassMapper import PredictionClassMapper
from classifier.PredictionClassMappers import BINARY_0_1, MULTICLASS_4


class PredictionModelPath(object):

    def __init__(self, 
                 nasdaqTag : list[str], 
                 tweetDataframePath : str, 
                 financialNumbersPath : str, 
                 wordvectorsPath : str, 
                 predictionModelPath : str,
                 top2vecModelPath : str,
                 isEqualSamplesForEachClass : bool, 
                 tweetGroupSize: int,
                 predictionClassMapper: PredictionClassMapper
                 ):
        self.nasdaqTag = nasdaqTag
        self.dataframePath = tweetDataframePath
        self.financialNumbersPath = financialNumbersPath
        self.wordvectorsPath = wordvectorsPath
        self.modelPath = predictionModelPath
        self.top2vecModelPath =  top2vecModelPath
        self.isEqualSamplesForEachClass = isEqualSamplesForEachClass
        self.tweetGroupSize = tweetGroupSize
        self.predictionClassMapper = predictionClassMapper
        
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
    
    def getPredictionClassMapper(self):
        return self.predictionClassMapper
    

    


AMAZON_REVENUE_5_LSTM_BINARY_CLASS = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN5',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    False,
    5,
    BINARY_0_1 
    )

TESLA_CAR_SALES_5_LSTM_BINARY_CLASS = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    5,
    BINARY_0_1 
    )


AMAZON_REVENUE_10_LSTM_BINARY_CLASS = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    10,
    BINARY_0_1 
    )

AMAZON_REVENUE_10_LSTM_MULTI_CLASS = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbersMulticlass.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN10Multiclass',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    10,
    MULTICLASS_4
    )

AMAZON_REVENUE_20_LSTM_MULTI_CLASS = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbersMulticlass.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20Multiclass',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    20,
    MULTICLASS_4
    )

TESLA_CAR_SALES_10_LSTM_BINARY_CLASS = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    10,
    BINARY_0_1
    )

TESLA_CAR_SALES_10_LSTM_MULTI_CLASS = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSalesMulticlass.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM10Multiclass',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    10,
    MULTICLASS_4
    )


APPLE__EPS_10_LSTM_BINARY_CLASS = PredictionModelPath(
    ["AAPL"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithEps.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleEpsLSTM10",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    True,
    10,
    BINARY_0_1
    )

APPLE__EPS_10_LSTM_MULTI_CLASS = PredictionModelPath(
    ["AAPL"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsAppleWithEpsMulticlass.csv",
    DataDirHelper().getDataDir()+"companyTweets\\appleEps.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\wordVectorsApple.txt",
    DataDirHelper().getDataDir() + "companyTweets\\model\\appleEpsLSTM10Multiclass",
    DataDirHelper().getDataDir() + "companyTweets\\appleTopicModel",
    True,
    10,
    MULTICLASS_4
    )


AMAZON_REVENUE_20_LSTM_BINARY_CLASS = PredictionModelPath(
    ["AMZN"],
    DataDirHelper().getDataDir()+"companyTweets\\amazonTweetsWithNumbers.csv",
    DataDirHelper().getDataDir()+"companyTweets\\amazonQuarterRevenue.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsAmazonV2.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\amazonRevenueLSTMN20',
    DataDirHelper().getDataDir() + "companyTweets\\amazonTopicModel",
    True,
    20,
    BINARY_0_1
    )

TESLA_CAR_SALES_20_LSTM_BINARY_CLASS = PredictionModelPath(
    ["TSLA"],
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsTeslaWithCarSales.csv",
    DataDirHelper().getDataDir()+"companyTweets\\teslaCarSales.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsTesla.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\teslaCarSalesLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\teslaTopicModel",
    True,
    20,
    BINARY_0_1
    )

GOOGLE_SE_MARKET_SHARE_5_LSTM_BINARY_CLASS = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM5',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    5,
    BINARY_0_1
    )

GOOGLE_SE_MARKET_SHARE_10_LSTM_BINARY_CLASS = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM10',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    10,
    BINARY_0_1
    )

GOOGLE_SE_MARKET_SHARE_20_LSTM_BINARY_CLASS = PredictionModelPath(
    ["GOOG"], 
    DataDirHelper().getDataDir()+"companyTweets\\CompanyTweetsGoogleWithSEMarketShare.csv",
    DataDirHelper().getDataDir()+"companyTweets\\googleSEMarketShare.csv",
    DataDirHelper().getDataDir()+ "companyTweets\\WordVectorsGoogle.txt",
    DataDirHelper().getDataDir() + 'companyTweets\\model\\googleWithSEMarketShareLSTM20',
    DataDirHelper().getDataDir() + "companyTweets\\googleTopicModel",
    True,
    20,
    BINARY_0_1
    )


        