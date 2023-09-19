'''
Created on 19.09.2023

@author: vital
'''
import pandas as pd
import matplotlib.pyplot as plt
from exploredata.FinancialFiguresExplore import FinancialFiguresExplore
from tweetpreprocess.DataDirHelper import DataDirHelper

class FinancialFiguresExplorePlots(object):
    '''
    classdocs
    '''


    def __init__(self, financialFiguresExplore: FinancialFiguresExplore):
        self.explore = financialFiguresExplore
    
    def createFiguresOverTimePlot(self,xLabel,yLabel,title):
        fig, ax = plt.subplots()
        dataframe = self.explore.getDataframe()
        for _, row in dataframe.iterrows():
            ax.bar(row[self.explore.fromDateColumnName], row[self.explore.valueColumnName], width=(row[self.explore.toDateColumnName] - row[self.explore.fromDateColumnName]).days, alpha=0.6, align='edge', label=f"{row['from_date'].strftime('%Y-%m-%d')} to {row['to_date'].strftime('%Y-%m-%d')}")
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.show()    
        
df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\teslaCarSales.csv')

plots = FinancialFiguresExplorePlots(FinancialFiguresExplore(df))
plots.createFiguresOverTimePlot("Time","Car sales numbers","Tesla car sales numbers over time")
