'''
Created on 27.01.2022

@author: vital
'''
import unittest
from tests import tweetnumbersconnectortest
from tests import nlpvectorstest
from tests import FiguresDiscretizerTest
from tests import HyperlinkRemoverTest
from tests import StopWordsFilterTest
from tests import TextFilterTest
from tests import TweetQueryTest
from tests import DateToTimestampTransformerTest
from tests import DateToTSPTest
from tests import PipelineTest
from tests import TweetDataframeSorterTest
from tests import FiguresIncreaseDecreaseClassCalculatorTest

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(tweetnumbersconnectortest))
suite.addTests(loader.loadTestsFromModule(nlpvectorstest))
suite.addTests(loader.loadTestsFromModule(FiguresDiscretizerTest))
suite.addTests(loader.loadTestsFromModule(HyperlinkRemoverTest))
suite.addTests(loader.loadTestsFromModule(StopWordsFilterTest))
suite.addTests(loader.loadTestsFromModule(TextFilterTest))
suite.addTests(loader.loadTestsFromModule(TweetQueryTest))
suite.addTests(loader.loadTestsFromModule(DateToTimestampTransformerTest))
suite.addTests(loader.loadTestsFromModule(DateToTSPTest))
suite.addTests(loader.loadTestsFromModule(PipelineTest))
suite.addTests(loader.loadTestsFromModule(TweetDataframeSorterTest))
suite.addTests(loader.loadTestsFromModule(FiguresIncreaseDecreaseClassCalculatorTest))
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)