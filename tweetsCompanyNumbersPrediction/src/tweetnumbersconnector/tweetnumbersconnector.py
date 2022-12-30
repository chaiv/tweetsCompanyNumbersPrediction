'''
Created on 22.01.2022

@author: vital
'''

class TweetNumbersConnector(object):
    '''
    Add tweets and economic figures together
    '''


    def __init__(self,fromTSPColumn = 'from_date',toTSPColumn ='to_date',valueColumn = 'value',postTSPColumn = 'post_date'):
        self.fromTSPColumn = fromTSPColumn
        self.toTSPColumn = toTSPColumn
        self.valueColumn = valueColumn
        self.postTSPColumn = postTSPColumn
        
    def getFiguresValue(self,allNumbersDf,postDate):
        return float(allNumbersDf.loc[(allNumbersDf[ self.fromTSPColumn] <= postDate) & (allNumbersDf[ self.toTSPColumn] >= postDate)][self.valueColumn])  
    def getTweetsWithNumbers(self,allTweetsDf, allNumbersDf):
        allTweetsWithNumbersDf =  allTweetsDf.copy()
        allTweetsWithNumbersDf[self.valueColumn] = allTweetsWithNumbersDf.apply(
            lambda x: self.getFiguresValue(allNumbersDf,x[self.postTSPColumn]),axis = 1
        )
        return  allTweetsWithNumbersDf
    
