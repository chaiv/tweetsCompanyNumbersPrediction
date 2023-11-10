'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import unittest
from featureinterpretation.ImportantWordsStore import ImportantWordStore,\
    createImportantWordStore, flatten_dict_lists, pad_dict_lists
from featureinterpretation.WordScoresWrapper import WordScoresWrapper
from nlpvectors.TweetGroup import TweetGroup


class WordScoresWrapperFake():
    
    def getSentenceIds(self):
        return [0,1]
    
    def getTokenIndexes(self):
        return [[1,2],[3,4,5]]
    
    def getTokens(self):
        return [["first","second"],["third","fourth","fifth"]]
    
    def getAttributions(self):
        return [[0.1,0.2],[0.3,0.4,0.5]]
    
    


class TestImportantWordStore(unittest.TestCase):
        
    def testTransformDict(self):
        data = {"id":[0,1],"class":[2,3]}
        transformed_data = pad_dict_lists(data)
        transformed_data = flatten_dict_lists(data)
        self.assertEqual([0,1],transformed_data["id"])
        self.assertEqual([2,3],transformed_data["class"])


    def testOneWordScoreWrapper(self):
        wordscoreWrappers =[WordScoresWrapperFake()]
        predictions = [1]
        result = createImportantWordStore(wordscoreWrappers,predictions)
        df = result.to_dataframe()
        self.assertEqual(5,len(df))
        self.assertEqual(0,df["id"].iloc[0])
        self.assertEqual(1,df["token_index"].iloc[0])
        self.assertEqual("first",df["token"].iloc[0])
        self.assertEqual(0.1,df["attribution"].iloc[0])
        self.assertEqual(1,df["prediction"].iloc[0])
   
   
   
        
    def testTwoWordScoreWrappers(self):
        
        sentenceIds = [576527445806870528, 576529132651196416, 576531462842949632, 576534180621729792, 576534748559863808]
        sentences =  ['For Options$AAPL$PFE$ORCL$MRK$NLY$COP$COH$DVN$NFX$TTWO$AMZNNice bottom', 'Stocks Trending Now:  $LL $LGND $BIOC $CSCO $ICA $USLV $ZIOP $CLDX $MHYS $ODP $JUNO $AMZN ~', 'A few more of interest: $AMBA $ADXS $AAL $AMGN $AMZN $ARIA $ARRY $ASHR ', 'Shares of Zulily down another 6 percent on Friday, slumping to all-time low () $ZU $AMZN $NILE ', '$XLI Investor Opinions Updated Friday, March 13, 2015 7:03:23 PM $INTC $AMZN $MCP $CSCO ']
        separatorIndexesInFeatureVector = [0, 15, 25, 35]
        totalFeatureVector = [158669, 3, 190, 138, 6962, 6078, 118, 29038, 6996, 3045, 4689, 6149, 2128, 2199, 0, 158669, 259, 804, 1322, 516, 388, 0, 1424, 1405, 1072, 158669, 26, 9823, 725, 149, 2542, 805, 76, 0, 7298, 158669, 1250, 85, 548, 51, 149, 711, 39, 0, 5694, 118]
        totalTokenIndexes =  [[], [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 2, 6, 8, 9, 11, 12, 15, 16], [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]]
        totalTokens = [[], ['stock', 'trend', 'now', 'lgnd', 'bioc', 'csco', 'ica', 'uslv', 'ziop', 'cldx', 'mhy', 'odp', 'juno', 'amzn'], ['interest', 'amba', 'adx', 'aal', 'amgn', 'amzn', 'aria', 'arri', 'ashr'], ['share', 'zulili', 'percent', 'fridai', 'slump', 'alltim', 'low', 'amzn', 'nile'], ['xli', 'investor', 'opinion', 'updat', 'fridai', 'march', 'intc', 'amzn', 'mcp', 'csco']]
        total_attributions = [[], [-0.0444437407106532, -0.00892199025111225, 0.026111857230962633, 0.0024328136071788743, -0.00957948556308604, -0.0059095074054248894, -0.018225511221042323, -0.072267520012524, -0.15621184774324054], [0.03283031441352818, -0.6306866131368114, -0.04114347288092393, -0.02937373506193071, -0.0007465558156013976, -0.002517684520844866, 0.011445869825630747, -0.0002673956181791256, 0.01861668874139369], [0.11482029916251385, 0.04854083575726556, 0.1115346873213904, 0.31006985189506925, 0.1704260018936845, 1.0, -0.04454184116032129, -0.023710538201927284, -0.34605599063591547]]
        prediction = 1            
        wordScoresWrapper1 = WordScoresWrapper(
                 TweetGroup(sentences=[],sentenceIds=sentenceIds,totalTokenIndexes= totalTokenIndexes,
                                 totalTokens=totalTokens,
                                 totalFeatureVector=totalFeatureVector,separatorIndexesInFeatureVector=separatorIndexesInFeatureVector,label = prediction),
                total_attributions)
        wordScoresWrapper2 = WordScoresWrapper(
                 TweetGroup(sentences=["", "another tweet1"],sentenceIds=[789,101],totalTokenIndexes=[[],[1,2]],
                                 totalTokens=[[], ["another", "tweet1"]],
                                 totalFeatureVector=[0,1,2],separatorIndexesInFeatureVector=[0],label = 0),
                [[], [0.8,0.9]])
        importantWordsStore = createImportantWordStore([wordScoresWrapper1,wordScoresWrapper2],[0,0])
        print(importantWordsStore.to_dataframe())
           
    
