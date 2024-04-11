'''
Created on 11.04.2024

@author: vital
'''

class TokenScoresSort(object):
    '''
    classdocs
    '''


    def __init__(self):
        pass
    
    def getSortedTokensWithScoresAsc(self, tokens, tokenScores):
        if len(tokens) != len(tokenScores):
            raise ValueError("tokens and tokenScores must be of the same length")
            
        # Pair the tokens with their scores and sort by the score
        token_score_pairs = list(zip(tokens, tokenScores))
        token_score_pairs.sort(key=lambda x: x[1])
    
        return token_score_pairs