'''
Created on 19.04.2026

@author: vital
'''
from enum import Enum

class TrainingMode(Enum):
    STRATIFIED_TEMPORAL = "stratified_temporal"
    SUBSEQUENT = "subsequent"
    TEMPORAL_SPLIT = "temporal_split"

    def isSortedTemporally(self):
        return self in (TrainingMode.STRATIFIED_TEMPORAL, TrainingMode.TEMPORAL_SPLIT)

