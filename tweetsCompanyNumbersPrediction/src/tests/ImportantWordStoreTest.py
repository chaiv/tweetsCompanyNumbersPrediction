'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import unittest
from featureinterpretation.ImportantWordsStore import ImportantWordStore,\
    createImportantWordStore, flatten_dict_lists, pad_dict_lists
from featureinterpretation.WordScoresWrapper import WordScoresWrapper


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
        
    # def test_toDfWithFirstNSortedByAttributionDfParam(self):
    #     df = pd.DataFrame({
    #         "id": ["id1","id2"],
    #         "token_index": [0,1],
    #         "token":["first","second"],
    #         "attribution": [0.5,0.6]
    #     })
    #     result = ImportantWordStore(None).toDfWithFirstNSortedByAttributionDfParam(1,df,"attribution",False)
    #     self.assertEqual(df.iloc[1]["attribution"],result.iloc[0]["attribution"])
    #
    #
    #
    # def test_toDfWithFirstNSortedByAttribution(self):
    #     store = ImportantWordStore({
    #             "id" : ['id1'],
    #             "token_index" : [[0, 1, 2, 3, 4]],
    #             "token" : [["this","is","the","first","tweet"]],
    #             "attribution" : [[0.1,0.2,0.3,0.4,0.5]]
    #             })  
    #     expected_data = [('id1', 4, 'tweet', 0.5), ('id1', 3, 'first', 0.4), ('id1', 2, 'the', 0.3), ('id1', 1, 'is', 0.2), ('id1', 0, 'this', 0.1)]
    #     actual_data = store.toDfWithFirstNSortedByAttribution(5, ascending = False).to_records(index=False)
    #     actual_data = [tuple(record) for record in actual_data]
    #     self.assertListEqual(list(actual_data), expected_data)
    #
    #
    # def test_to_dataframe(self):
    #     store = ImportantWordStore({
    #             "id" : [1,2],
    #             "token_index" : [[0, 1, 2, 3, 4, 5],[0, 1, 2, 3]],
    #             "token" : [["the", "cat", "is", "on", "the", "mat"],["the", "dog", "is", "outside"]],
    #             "attribution" : [[0.1, -0.2, 0.3, 0.05, -0.1, 0.15],[-0.05, 0.15, 0.2, 0.1]],
    #             "prediction" : [0,1]
    #             })        
    #     df = store.to_dataframe()
    #     self.assertEqual(list(df.columns), ['id', 'token_index','token', 'attribution',"prediction"])
    #     expected_data = [
    #         (1, 0,"the", 0.1,0),
    #         (1, 1,"cat", -0.2,0),
    #         (1, 2,"is", 0.3,0),
    #         (1, 3,"on", 0.05,0),
    #         (1, 4,"the", -0.1,0),
    #         (1, 5, "mat", 0.15,0),
    #         (2, 0,"the", -0.05,1),
    #         (2, 1,"dog", 0.15,1),
    #         (2, 2,"is", 0.2,1),
    #         (2, 3,"outside", 0.1,1)
    #     ]
    #     for row, expected_row in zip(df.itertuples(index=False), expected_data):
    #         self.assertEqual(tuple(row), expected_row)
    #

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

           
    
