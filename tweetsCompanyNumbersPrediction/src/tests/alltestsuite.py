'''
Created on 27.01.2022

@author: vital
'''
import unittest
from tests import tweetnumbersconnectortest, TestTopicHeaderAddToDataframe,\
    TweetDataframeExploreTest
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
from tests import FeatureDataframeCreatorTest
from tests import TweetTextFilterTransformerTest
from tests import TestPredictor
from tests import EqualClassSamplerTest
from tests import ImportantWordStoreTest
from tests import ClassificationMetricsTest


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
suite.addTests(loader.loadTestsFromModule(FeatureDataframeCreatorTest))
suite.addTests(loader.loadTestsFromModule(EqualClassSamplerTest))
suite.addTests(loader.loadTestsFromModule(TweetTextFilterTransformerTest))
suite.addTests(loader.loadTestsFromModule(ImportantWordStoreTest))
suite.addTests(loader.loadTestsFromModule(ClassificationMetricsTest))
suite.addTests(loader.loadTestsFromModule(TestPredictor))
suite.addTests(loader.loadTestsFromModule(TestTopicHeaderAddToDataframe))
suite.addTests(loader.loadTestsFromModule(TweetDataframeExploreTest))
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)