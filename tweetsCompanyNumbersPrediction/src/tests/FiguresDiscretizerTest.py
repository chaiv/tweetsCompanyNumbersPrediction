'''
Created on 01.02.2022

@author: vital
'''
import unittest
import pandas as pd
from tweetpreprocess.FiguresPercentChangeCalculator import FiguresPercentChangeCalculator
from tweetpreprocess.FiguresDiscretizer import FiguresDiscretizer

class FiguresDiscretizerTest(unittest.TestCase):


    def testFiguresDiscretizer(self):
        figures =  pd.DataFrame(
                  [
                  ("01/10/2014", "31/12/2014",0.95700),    
                      
                  ("01/01/2015", "31/03/2015",0.93988),
                  ("01/04/2015", "30/06/2015",1.20 ),
                  ("01/07/2015", "30/09/2015",0.96380),
                  ("01/10/2015", "31/12/2015",1.75),
                  
                  ("01/01/2016", "31/03/2016",1.60),
                  ("01/04/2016", "30/06/2016",1.56 ),
                  ("01/07/2016", "30/09/2016",2.30 ),
                  ("01/10/2016", "31/12/2016",2.29 ),
                  
                  ( "01/01/2017", "31/03/2017",2.70),
                  ("01/04/2017", "30/06/2017",2.79),
                  ("01/07/2017", "30/09/2017",2.99),
                  ("01/10/2017", "31/12/2017",3.29),
                  
                  ("01/01/2018", "31/03/2018",3.41),
                  ("01/04/2018", "30/06/2018",4.00),
                  ("01/07/2018", "30/09/2018",6.82),
                  ("01/10/2018", "31/12/2018",7.23),
                  
                  ("01/01/2019", "31/03/2019",4.54),
                  ("01/04/2019", "30/06/2019",6.35),
                  ("01/07/2019", "30/09/2019",6.30), 
                  ("01/10/2019", "31/12/2019",7.38),
                  
                  ("01/01/2020", "31/03/2020",5.99),
                  ("01/04/2020", "30/06/2020",6.04),
                  ("01/07/2020", "30/09/2020",8.77),
                  ("01/10/2020", "31/12/2020",10.70)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        percentChangeCalculator = FiguresPercentChangeCalculator ("value")
        discretizer = FiguresDiscretizer(percentChangeCalculator.getFiguresWithClasses(figures),'percentChange', 5)
        print(discretizer.getFiguresWithClasses())

    def testFiguresPercentChangeCalculator(self):
        figures =  pd.DataFrame(
                  [
                  ("01/10/2014", "31/12/2014",0.95700),    
                      
                  ("01/01/2015", "31/03/2015",0.93988),
                  ("01/04/2015", "30/06/2015",1.20 ),
                  ("01/07/2015", "30/09/2015",0.96380),
                  ("01/10/2015", "31/12/2015",1.75),
                  
                  ("01/01/2016", "31/03/2016",1.60),
                  ("01/04/2016", "30/06/2016",1.56 ),
                  ("01/07/2016", "30/09/2016",2.30 ),
                  ("01/10/2016", "31/12/2016",2.29 ),
                  
                  ( "01/01/2017", "31/03/2017",2.70),
                  ("01/04/2017", "30/06/2017",2.79),
                  ("01/07/2017", "30/09/2017",2.99),
                  ("01/10/2017", "31/12/2017",3.29),
                  
                  ("01/01/2018", "31/03/2018",3.41),
                  ("01/04/2018", "30/06/2018",4.00),
                  ("01/07/2018", "30/09/2018",6.82),
                  ("01/10/2018", "31/12/2018",7.23),
                  
                  ("01/01/2019", "31/03/2019",4.54),
                  ("01/04/2019", "30/06/2019",6.35),
                  ("01/07/2019", "30/09/2019",6.30), 
                  ("01/10/2019", "31/12/2019",7.38),
                  
                  ("01/01/2020", "31/03/2020",5.99),
                  ("01/04/2020", "30/06/2020",6.04),
                  ("01/07/2020", "30/09/2020",8.77),
                  ("01/10/2020", "31/12/2020",10.70)
                  ],
                  columns=["from_date","to_date","value"]
                  )
        percentChangeCalculator = FiguresPercentChangeCalculator ("value")
        figuresWithPercentChanges = percentChangeCalculator.getFiguresWithClasses(figures)
        self.assertIsNotNone(figuresWithPercentChanges.iloc[0]["value"])
        self.assertEqual(figuresWithPercentChanges.iloc[1]["value"]/figuresWithPercentChanges.iloc[0]["value"],figuresWithPercentChanges.iloc[1]["percentChange"])
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()