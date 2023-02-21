'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd

class TokenAttributionStore:
    def __init__(self):
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
                rows.append({"id": id_value, "token": token, "token_index": token_index, "attribution": attribution})
        return pd.DataFrame(rows, columns=["id", "token_index","token", "attribution"])