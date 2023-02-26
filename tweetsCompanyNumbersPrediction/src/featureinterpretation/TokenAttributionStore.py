'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd

class TokenAttributionStore:
    def __init__(self, idColumnName = "id", tokenColumnName = "token", tokenIndexColumnName = "token_index", atrributionColumnName = "attribution"):
        self.idColumnName = idColumnName
        self.tokenColumnName = tokenColumnName
        self.tokenIndexColumnName = tokenIndexColumnName
        self.atrributionColumnName = atrributionColumnName
        self.data = {}

    def add_data(self, id_value, token_indexes, tokens, attributions):
        attributions = attributions[:len(tokens)]
        self.data[id_value] = {"tokens": tokens, "token_indexes": token_indexes, "attributions": attributions}

    def add_multiple_data(self, data):
        for item in data:
            id_value, token_indexes, tokens, attributions = item
            self.add_data(id_value, token_indexes, tokens, attributions)


    def to_dataframe(self):
        rows = []
        for id_value, value in self.data.items():
            tokens = value["tokens"]
            token_indexes = value["token_indexes"]
            attributions = value["attributions"]
            for token, token_index, attribution in zip(tokens, token_indexes, attributions):
                rows.append({self.idColumnName: id_value, self.tokenColumnName: token,  self.tokenIndexColumnName: token_index, self.atrributionColumnName: attribution})
        return pd.DataFrame(rows, columns=[self.idColumnName, self.tokenIndexColumnName,self.tokenColumnName,self.atrributionColumnName ])