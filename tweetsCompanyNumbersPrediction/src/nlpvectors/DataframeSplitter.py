from nlpvectors.VocabularyCreator import SEP_TOKEN
'''
Created on 24.04.2023

@author: vital
'''
import pandas as pd

class DataframeSplitter(object):


    def __init__(self):
        pass
    
    def getRowIndexesOfSplitsAsFlattenedList(self,splits,splitIndexes):
        row_indexes= []
        for split_index in splitIndexes:
            row_indexes.extend(splits[split_index ])
        return row_indexes
    
    
    def getDfWithGroupedTweets(self,df,split_size,idColumnName = "tweet_id",bodyColumnName="body",classColumnName="class",
                               combinedIdsColumnName= 'tweet_ids',combinedBodyColumnName='body'
                               ):
        splits = self.getDfSplitIndexes(df, split_size,idColumnName, classColumnName)
        combined_text_lists = []
        combined_ids_lists = []
        combined_class_lists = []
        for split in splits:
            combined_text = df.loc[df['tweet_id'].isin(split), bodyColumnName].str.cat(sep=SEP_TOKEN)
            combined_ids = split
            combined_class = df[df[idColumnName]==split[0]].iloc[0][classColumnName]
            combined_text_lists.append(combined_text)
            combined_ids_lists.append(combined_ids)
            combined_class_lists.append( combined_class)
        grouped_tweets_df = pd.DataFrame({combinedIdsColumnName: combined_ids_lists,combinedBodyColumnName: combined_text_lists, classColumnName : combined_class_lists })
        return grouped_tweets_df
    
    def getDfSplitIndexes(self, df, split_size,idColumnName = "tweet_id", classColumnName="class"):
        # Create an empty list to store the resulting splits
        splits = []
        
        # Get the unique classes in the DataFrame
        unique_classes = df[classColumnName].unique()
        
        # Iterate through each unique class
        for unique_class in unique_classes:
            # Filter the DataFrame to keep only the rows with the current class
            class_df = df[df[classColumnName] == unique_class]

            # Calculate the number of splits for the current class
            num_splits = len(class_df) // split_size

            # Add the splits to the list
            for i in range(num_splits):
                splitDf = class_df.iloc[i * split_size : (i + 1) * split_size]
                splits.append(splitDf[idColumnName].tolist())

            # Add the remaining rows to a smaller split if there are any
            remaining_rows = len(class_df) % split_size
            if remaining_rows > 0:
                splitDf = class_df.iloc[-remaining_rows:]
                splits.append(splitDf[idColumnName].tolist())

        return splits    
         
        