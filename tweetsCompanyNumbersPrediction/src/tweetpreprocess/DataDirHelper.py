'''
Created on 15.01.2023

@author: vital
'''

from os import path
class DataDirHelper(object):

    '''
    classdocs
    '''


    def __init__(self,primaryDir='C:\\Users\\vital\\Google Drive\\promotion\\', secondaryDir=r"G:\\Meine Ablage\\promotion\\"):
        self.primaryDir = primaryDir
        self.secondaryDir = secondaryDir

    def getDataDir(self):
        if(path.exists(self.primaryDir)):
            return  self.primaryDir
        if (path.exists(self.secondaryDir)):
            return  self.secondaryDir
        raise Exception("No data directory found!")