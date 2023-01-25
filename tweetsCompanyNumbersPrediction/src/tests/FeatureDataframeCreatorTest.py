'''
Created on 15.01.2023

@author: vital
'''
import unittest
import pandas as pd
from tests.FeatureVectorMapperFake import FeatureVectorMapperFake
from nlpvectors.FeatureDataframeCreator import FeatureDataframeCreator


class FeatureDataframeCreatorTest(unittest.TestCase):


    def testCreateDf(self):
        tweets =  pd.DataFrame(
                [
                  ("1","1",1),
                  ("2","2",0),
                  ("3","3",1),
                  ],
                  columns=["tweet_id","post_date","class"]
                )
        
        mapper = FeatureVectorMapperFake([
                [1,2,3],
                [4,5,6],
                [8,9,10]
            ])
        result = FeatureDataframeCreator(mapper).createFeatureDataframe(tweets)
        print(result)
        firstvector=result["features"].iloc[0]
        self.assertEqual(firstvector,[1,2,3] )
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCreateDf']
    unittest.main()