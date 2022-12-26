'''
Created on 26.12.2022

@author: vital
'''
import unittest
import pandas as pd

class Pipeline(unittest.TestCase):


    def testWhenAllPipelineStepsThenCompletedDf(self):
        
        tweets = pd.DataFrame(
                  [
                  ("1","Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur","company1"),
                  ("2","company next charger will automatically connect to your car http://flip.it/dummylink","company2"),
                  ("3","Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink","company1"),
                  ("4","Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink","company2"),
                  ("5","Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink","company1")
                  ],
                  columns=["tweet_id","body","ticker_symbol"]
                  )
        
        
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()