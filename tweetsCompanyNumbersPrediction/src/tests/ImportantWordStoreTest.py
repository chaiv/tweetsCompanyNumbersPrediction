'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd
import unittest
from featureinterpretation.ImportantWordsStore import ImportantWordStore


class TestImportantWordStore(unittest.TestCase):
        
    def test_toDfWithFirstNSortedByAttributionDfParam(self):
        df = pd.DataFrame({
            "id": ["id1","id2"],
            "token_index": [0,1],
            "token":["first","second"],
            "attribution": [0.5,0.6]
        })
        result = ImportantWordStore(None).toDfWithFirstNSortedByAttributionDfParam(1,df,"attribution",False)
        self.assertEqual(df.iloc[1]["attribution"],result.iloc[0]["attribution"])
        
    
    
    def test_toDfWithFirstNSortedByAttribution(self):
        store = ImportantWordStore({
                "id" : ['id1'],
                "token_index" : [[0, 1, 2, 3, 4]],
                "token" : [["this","is","the","first","tweet"]],
                "attribution" : [[0.1,0.2,0.3,0.4,0.5]]
                })  
        expected_data = [('id1', 4, 'tweet', 0.5), ('id1', 3, 'first', 0.4), ('id1', 2, 'the', 0.3), ('id1', 1, 'is', 0.2), ('id1', 0, 'this', 0.1)]
        actual_data = store.toDfWithFirstNSortedByAttribution(5, ascending = False).to_records(index=False)
        actual_data = [tuple(record) for record in actual_data]
        self.assertListEqual(list(actual_data), expected_data)
    
    def test_to_dataframe(self):
        store = ImportantWordStore({
                "id" : [1,2],
                "token_index" : [[0, 1, 2, 3, 4, 5],[0, 1, 2, 3, 4, 5]],
                "token" : [["the", "cat", "is", "on", "the", "mat"],["the", "dog", "is", "in", "the", "yard"]],
                "attribution" : [[0.1, -0.2, 0.3, 0.05, -0.1, 0.15],[-0.05, 0.15, 0.2, 0.1, -0.1, 0.1]]
                })        
        df = store.to_dataframe()
        self.assertEqual(list(df.columns), ['id', 'token_index','token', 'attribution'])
        expected_data = [
            (1, 0,"the", 0.1),
            (1, 1,"cat", -0.2),
            (1, 2,"is", 0.3),
            (1, 3,"on", 0.05),
            (1, 4,"the", -0.1),
            (1, 5, "mat", 0.15),
            (2, 0,"the", -0.05),
            (2, 1,"dog", 0.15),
            (2, 2,"is", 0.2),
            (2, 3,"in", 0.1),
            (2, 4,"the", -0.1),
            (2, 5, "yard", 0.1)
        ]
        for row, expected_row in zip(df.itertuples(index=False), expected_data):
            self.assertEqual(tuple(row), expected_row)