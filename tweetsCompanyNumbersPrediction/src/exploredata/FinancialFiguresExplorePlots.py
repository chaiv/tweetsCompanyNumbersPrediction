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
    

        
    def createFiguresOverTimePlot(self, xLabel, yLabel, title):
        """
        Creates a plot with lines from the x-axis to points and x-axis values as quarters.

        Args:
            xLabel (str): Label for the x-axis.
            yLabel (str): Label for the y-axis.
            title (str): Title of the plot.
        """
        fig, ax = plt.subplots()
        dataframe = self.explore.getDataframe()

        # Prepare the data
        quarters = []
        for _, row in dataframe.iterrows():
            midpoint = row[self.explore.fromDateColumnName] + (row[self.explore.toDateColumnName] - row[self.explore.fromDateColumnName]) / 2
            quarter = f"Q{(midpoint.month - 1) // 3 + 1}/{str(midpoint.year)[-2:]}"
            quarters.append(quarter)
            # Plot vertical line and point
            ax.plot([quarter, quarter], [0, row[self.explore.valueColumnName]], 'k-', alpha=0.7)  # Line
            ax.scatter(quarter, row[self.explore.valueColumnName], color='black', zorder=5)  # Point

        # Set x-axis labels
        ax.set_xticks(quarters)
        ax.set_xticklabels(quarters, rotation=45, ha='right',fontsize=14)

        # Set labels and title
        plt.xlabel(xLabel,fontsize=14)
        plt.ylabel(yLabel,fontsize=14)
        #plt.title(title,fontsize=14)

        # Show the plot
        plt.tight_layout()
        plt.show()    
        
#df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\amazonQuarterRevenue.csv')        
#df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\teslaCarSales.csv')
df =  pd.read_csv(DataDirHelper().getDataDir()+ 'companyTweets\\appleEps.csv')

plots = FinancialFiguresExplorePlots(FinancialFiguresExplore(df))
plots.createFiguresOverTimePlot("Quarter","Metrics","")
