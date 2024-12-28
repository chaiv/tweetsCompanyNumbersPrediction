'''
Created on 17.12.2024

@author: vital
'''

import pandas as pd

class FinancialFiguresClassifier:
    def __init__(self, classes, percentChangeDfColumn = 'percent_change', classColumnName = "multi_class"):
        """
        Initialize the classifier with user-defined classes and their percent ranges.

        Parameters:
        - classes (list of dict): List of dictionaries where each dictionary represents a class.
          Format: {"class_name": int, "from": float, "to": float}
          Example: [
              {"class_name": 0, "from": -float('inf'), "to": -0.01},
              {"class_name": 1, "from": 0, "to": 10},
              {"class_name": 2, "from": 10, "to": 20},
              {"class_name": 3, "from": 20, "to": float('inf')}
          ]
        """
        self.classes = classes
        self.percentChangeDfColumn = percentChangeDfColumn
        self.classColumnName = classColumnName
        self.validate_classes()

    def validate_classes(self):
        """
        Validate that the input classes are in the correct format.
        Each class should be a dictionary with keys: 'class_name', 'from', 'to'.
        """
        required_keys = {"class_name", "from", "to"}
        if not isinstance(self.classes, list):
            raise ValueError("Classes should be a list of dictionaries.")
        for class_def in self.classes:
            if not isinstance(class_def, dict):
                raise ValueError(f"Each class definition must be a dictionary. Found: {type(class_def)}")
            if not required_keys.issubset(class_def.keys()):
                raise ValueError(f"Each class must contain the keys: {required_keys}. Found: {class_def.keys()}")
            if not isinstance(class_def["class_name"], int):
                raise ValueError("'class_name' must be an integer.")
            if not isinstance(class_def["from"], (int, float)) or not isinstance(class_def["to"], (int, float)):
                raise ValueError("'from' and 'to' must be numeric values.")

    def classify(self, value):
        for class_def in self.classes:
            if class_def["from"] <= value <= class_def["to"]:
                return class_def["class_name"]
        return None
    
    def calculate_counts(self,df):
        counts = {class_def["class_name"]: 0 for class_def in self.classes}
        for class_name, count in df[self.classColumnName].value_counts().items():
            counts[class_name] = count
        return counts

    def add_classes(self, df):
        df[ self.classColumnName] = df[self.percentChangeDfColumn].apply(self.classify)
