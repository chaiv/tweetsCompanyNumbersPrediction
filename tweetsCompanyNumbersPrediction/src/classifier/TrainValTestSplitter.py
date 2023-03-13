'''
Created on 12.03.2023

@author: vital
'''

class TrainValTestSplitter:
      

    def split(self, input_df):
        #train_val, test = train_test_split(df, random_state=1337, test_size=0.3)
        #train, val = train_test_split(train_val, random_state=1337, test_size=0.3)
        '''
            Because the tweet data and quarter numbers have relevant changes over time, the data is not randomly selected
        '''
        num_rows = len(input_df)
        train_end = int(num_rows * 0.6)  # 60% of the data
        val_end = int(num_rows * 0.8)  # 20% validation data
        train_df = input_df[:train_end]
        val_df = input_df[train_end:val_end]
        test_df = input_df[val_end:]
        return train_df, val_df, test_df
            