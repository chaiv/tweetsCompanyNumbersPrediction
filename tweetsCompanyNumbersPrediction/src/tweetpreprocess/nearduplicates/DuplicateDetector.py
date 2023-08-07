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
        exact_duplicates_mask = self.dataframe.duplicated(self.bodyColumnName, keep=False)    
        exact_duplicates_df = self.dataframe[exact_duplicates_mask]
        return exact_duplicates_df
    
        