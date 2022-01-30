'''
Created on 27.01.2022

@author: vital
'''
import unittest
from tests import tweetnumbersconnectortest
from tests import nlpvectorstest

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(tweetnumbersconnectortest))
suite.addTests(loader.loadTestsFromModule(nlpvectorstest))
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)