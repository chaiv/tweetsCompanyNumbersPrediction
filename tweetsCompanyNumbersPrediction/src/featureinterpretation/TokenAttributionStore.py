'''
Created on 17.02.2023

@author: vital
'''
import pandas as pd

class TokenAttributionStore:
    def __init__(self):
        self.data = {}

    def add_data(self, id_value, tokens, attributions):
        if len(tokens) != len(attributions):
            raise ValueError("The length of the tokens and attributions lists must be the same.")
        self.data[id_value] = {"tokens": tokens, "attributions": attributions}

    def to_dataframe(self):
        rows = []
        for id_value, value in self.data.items():
            tokens = value["tokens"]
            attributions = value["attributions"]
            for token, attribution in zip(tokens, attributions):
                rows.append({"id": id_value, "token": token, "attribution": attribution})
        return pd.DataFrame(rows, columns=["id", "token", "attribution"])