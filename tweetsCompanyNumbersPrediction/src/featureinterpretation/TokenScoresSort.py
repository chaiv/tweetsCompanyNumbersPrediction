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
    
    def getSortedTokensAndScoresDescFromListOfLists(self, tokens : list[list[str]], tokenScores: list[list[float]]):
        tokens_flattened = [item for sublist in tokens for item in sublist]
        scores_flattened = [item for sublist in tokenScores for item in sublist]
        return self.getSortedTokensAndScoresDesc( tokens_flattened, scores_flattened )
    
    def getSortedTokensAndScoresAscFromListOfLists(self,tokens : list[list[str]], tokenScores: list[list[float]]):
        sorted_tokens_asc, sorted_scores_asc = self.getSortedTokensAndScoresDescFromListOfLists(tokens,tokenScores)
        sorted_tokens_asc.reverse()
        sorted_scores_asc.reverse()
        return  sorted_tokens_asc, sorted_scores_asc
        

    def getSortedTokensAndScoresDesc(self, tokens, tokenScores):
        sorted_tokens_asc, sorted_scores_asc = self.getSortedTokensAndScoresAsc(tokens, tokenScores)
        sorted_tokens_asc.reverse()
        sorted_scores_asc.reverse()
        return sorted_tokens_asc, sorted_scores_asc
    
    def getAsSingleList(self,tokens, tokenScores):
        return list(zip(tokens, tokenScores))
    
    
    def getSortedTokensAndScoresAsc(self, tokens, tokenScores):
        #return sorted tokens and token Scores lists sorted ascending by token scores
        if len(tokens) != len(tokenScores):
            raise ValueError("tokens and tokenScores must be of the same length")

        # Pair the tokens with their scores and sort by the score
        token_score_pairs = list(zip(tokens, tokenScores))
        token_score_pairs.sort(key=lambda x: x[1])

        # Unzip the sorted pairs into two lists
        sorted_tokens, sorted_scores = zip(*token_score_pairs)
        return list(sorted_tokens), list(sorted_scores)