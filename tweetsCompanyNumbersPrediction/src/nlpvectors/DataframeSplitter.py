'''
Created on 24.04.2023

@author: vital
'''

class DataframeSplitter(object):


    def __init__(self):
        pass
    
    def getRowIndexesOfSplitsAsFlattenedList(self,splits,splitIndexes):
        row_indexes= []
        for split_index in splitIndexes:
            row_indexes.extend(splits[split_index ])
        return row_indexes
    
    def getDfSplitIndexes(self, df, split_size, classColumnName="class"):
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
                splits.append(splitDf.index.tolist())

            # Add the remaining rows to a smaller split if there are any
            remaining_rows = len(class_df) % split_size
            if remaining_rows > 0:
                splitDf = class_df.iloc[-remaining_rows:]
                splits.append(splitDf.index.tolist())

        return splits    
         
        