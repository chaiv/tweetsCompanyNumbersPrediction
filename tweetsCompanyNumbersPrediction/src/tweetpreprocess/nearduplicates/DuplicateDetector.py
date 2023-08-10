'''
Created on 07.08.2023

@author: vital
'''

class DuplicateDetector(object):



    def __init__(self, dataframe, 
                 bodyColumnName = "body"):
        self.dataframe = dataframe
        self.bodyColumnName = bodyColumnName

    def getDuplicatesDataframe(self):  
        return self.getDuplicatesDfWithOriginalRows().drop_duplicates(subset=self.bodyColumnName, keep='first')  
    
    def getDuplicatesDfWithOriginalRows(self):
        exact_duplicates_mask = self.dataframe.duplicated(self.bodyColumnName, keep=False)    
        exact_duplicates_df = self.dataframe[exact_duplicates_mask]
        return exact_duplicates_df
        
    
    def getDataframeWithoutDuplicates(self):
        df = self.dataframe.drop_duplicates(subset=self.bodyColumnName, keep='first') 
        df_reset_index = df.reset_index(drop=True)
        return  df_reset_index    