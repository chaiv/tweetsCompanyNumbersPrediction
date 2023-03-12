'''
Created on 12.03.2023

@author: vital
'''

import numpy as np

class TrainValTestSplitter:
    def __init__(self, test_frac=0.3, val_frac=0.3):
        self.test_frac = test_frac
        self.val_frac = val_frac
        
        
    '''
    Because the tweet data and quarter numbers have relevant changes over time, the data is not shuffled but splitted on random rows
    '''
    def split(self, input_df):
        n_rows = len(input_df)
        idx = np.random.randint(n_rows)
        
        df2 = input_df.iloc[idx:int(idx+0.3*n_rows)]
        train_df = input_df.drop(df2.index)
        
        n_rows = len(df2)
        idx = np.random.randint(n_rows)
        
        test_df = df2.iloc[idx:int(idx+0.3*n_rows)]
        val_df = df2.drop(test_df.index)
        
        return train_df, val_df, test_df
        