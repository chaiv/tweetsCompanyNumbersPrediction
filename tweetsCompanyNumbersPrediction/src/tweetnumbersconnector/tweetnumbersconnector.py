'''
Created on 22.01.2022

@author: vital
'''

class TweetNumbersConnector(object):
    '''
    Add tweets and economic figures together
    '''


    def __init__(self,fromDateColumn = 'from_date',toDateColumn ='to_date',valueColumn = 'value',postDateColumn = 'post_date'):
        self.fromDateColumn = fromDateColumn
        self.toDateColumn = toDateColumn
        self.valueColumn = valueColumn
        self.postDateColumn = postDateColumn
        
    def getFiguresValue(self,allNumbersDf,postDate):
        return float(allNumbersDf.loc[(allNumbersDf[ self.fromDateColumn] <= postDate) & (allNumbersDf[ self.toDateColumn] >= postDate)][self.valueColumn])  
    def getTweetsWithNumbers(self,allTweetsDf, allNumbersDf):
        allTweetsWithNumbersDf =  allTweetsDf.copy()
        allTweetsWithNumbersDf['value'] = allTweetsWithNumbersDf.apply(
            lambda x: self.getFiguresValue(allNumbersDf,x[self.postDateColumn]),axis = 1
        )
        return  allTweetsWithNumbersDf
    
