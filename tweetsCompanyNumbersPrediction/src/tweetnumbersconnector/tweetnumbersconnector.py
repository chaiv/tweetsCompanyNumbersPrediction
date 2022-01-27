'''
Created on 22.01.2022

@author: vital
'''

class TweetNumbersConnector(object):
    '''
    Add tweets and economic figures together
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    def getFiguresValue(self,allNumbersDf,postDate):
        return float(allNumbersDf.loc[(allNumbersDf['from_date'] <= postDate) & (allNumbersDf['to_date'] >= postDate)]['value'])  
    def getTweetsWithNumbers(self,allTweetsDf, allNumbersDf):
        allTweetsWithNumbersDf =  allTweetsDf.copy()
        allTweetsWithNumbersDf['value'] = allTweetsWithNumbersDf.apply(
            lambda x: self.getFiguresValue(allNumbersDf,x["post_date"]),axis = 1
        )
        return  allTweetsWithNumbersDf
    
