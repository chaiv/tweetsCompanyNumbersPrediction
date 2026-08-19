from nlpvectors.VocabularyCreator import SEP_TOKEN
'''
Created on 24.04.2023
The purpose of this class is to split a dataframe into group of tweets that have the same class value. 
Because of pandas large data perfomance issues the splits are returned as list of indexes or tweet ids
@author: vital
'''
import pandas as pd
from collections import Counter

class DataframeSplitter(object):


    def __init__(self):
        pass
    
    def getClassCountsOfSplitsByIndexes(self,df,splits,splitIndexes,idColumnName = "tweet_id",classColumnName="class"):
        splitsFromIndexes = []
        for splitIndex in splitIndexes:
            splitsFromIndexes.append(splits[splitIndex])
        return self.getClassCountsOfSplits(df, splitsFromIndexes,idColumnName,classColumnName)
    
    
    def getClassCountsOfSplits(self,df,splits,idColumnName = "tweet_id",classColumnName="class"):
        classCounts = Counter()
        for split in splits:
             classLabel = df[df[idColumnName]==split[0]].iloc[0][classColumnName]
             classCounts[classLabel] += 1
        return  classCounts   
        
    
    
    def getIdsOfSplitsAsFlattenedList(self,splits,splitIndexes):
        ids= []
        for split_index in splitIndexes:
            ids.extend(splits[split_index ])
        return ids
    
    
    def getDfWithGroupedTweets(self,df,split_size,idColumnName = "tweet_id",bodyColumnName="body",classColumnName="class",
                               combinedIdsColumnName= 'tweet_ids',combinedBodyColumnName='body'
                               ):
        splits = self.getSplitIds(df, split_size,idColumnName, classColumnName)
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
    
    def getSplitIds(self, df, split_size,idColumnName = "tweet_id", classColumnName="class"):
        # NOTE: every group is built from consecutive tweets OF THE SAME CLASS, so constructing the
        # inputs already requires the target label. On unlabeled data, as at prediction time, such
        # groups cannot be formed; getSplitIdsByTime below groups without consulting the label.
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
                splitIds = splitDf[idColumnName].tolist() 
                splits.append(splitIds)

            # Add the remaining rows to a smaller split if there are any
            remaining_rows = len(class_df) % split_size
            if remaining_rows > 0:
                splitDf = class_df.iloc[-remaining_rows:]
                splitIds = splitDf[idColumnName].tolist() 
                splits.append(splitIds)

        return splits

    def getSplitIdsByTime(self, df, split_size, idColumnName="tweet_id",
                          periodColumnName=None):
        """Group consecutive rows without consulting the target class.

        The dataframe order is treated as chronological.  When ``periodColumnName`` is supplied,
        groups are restarted at every reporting-period boundary; this is the safe training form for
        quarter-constant targets.  Without it, the result is suitable for unlabeled inference but a
        group may cross a target-period boundary and must not simply inherit its first row's label.
        """
        splits = []
        dataframes = [df]
        if periodColumnName is not None:
            dataframes = [periodDf for _, periodDf in df.groupby(periodColumnName, sort=False)]
        for periodDf in dataframes:
            for i in range(0, len(periodDf), split_size):
                splits.append(periodDf.iloc[i:i + split_size][idColumnName].tolist())
        return splits
