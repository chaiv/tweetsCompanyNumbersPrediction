'''
Created on 27.01.2022

@author: vital
'''
import unittest
from tests import tweetnumbersconnectortest

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(tweetnumbersconnectortest))
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)