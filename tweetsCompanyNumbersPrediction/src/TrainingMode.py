'''
Created on 19.04.2026

@author: vital
'''
from enum import Enum

class TrainingMode(Enum):
    STRATIFIED_TEMPORAL = "stratified_temporal"
    SUBSEQUENT = "subsequent"
    TEMPORAL_SPLIT = "temporal_split"
    STRATIFIED_KFOLD_TEMPORAL_PER_CLASS = "stratified_kfold_temporal_per_class"

    def isSortedTemporally(self):
        return self in (TrainingMode.STRATIFIED_TEMPORAL, TrainingMode.TEMPORAL_SPLIT, TrainingMode.STRATIFIED_KFOLD_TEMPORAL_PER_CLASS)

