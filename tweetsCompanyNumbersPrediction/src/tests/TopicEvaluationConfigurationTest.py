'''Regression tests for the topic-coherence metric configuration.'''
import unittest
from unittest.mock import patch

from topicmodelling.TopicEvaluation import TopicEvaluation


class _TopicModelStub(object):

    def getTopicWordsScoresAndIds(self):
        return [['battery', 'vehicle']], None, None

    def get_documents(self):
        return ['battery vehicle']


class _TokenizerStub(object):

    def tokenize(self, document):
        return document.split()


class TopicEvaluationConfigurationTest(unittest.TestCase):

    @patch('topicmodelling.TopicEvaluation.CoherenceModel')
    def testHistoricalGensimDefaultIsNowExplicit(self, coherenceModel):
        coherenceModel.return_value.get_coherence.return_value = 0.42
        evaluation = TopicEvaluation(_TopicModelStub(), _TokenizerStub())

        self.assertEqual(0.42, evaluation.get_topic_coherence())
        self.assertEqual('c_v', coherenceModel.call_args.kwargs['coherence'])

    @patch('topicmodelling.TopicEvaluation.CoherenceModel')
    def testUciCoherenceCanBeRequestedExplicitly(self, coherenceModel):
        coherenceModel.return_value.get_coherence.return_value = 0.11
        evaluation = TopicEvaluation(_TopicModelStub(), _TokenizerStub(), coherenceMeasure='c_uci')

        self.assertEqual(0.11, evaluation.get_topic_coherence())
        self.assertEqual('c_uci', coherenceModel.call_args.kwargs['coherence'])


if __name__ == '__main__':
    unittest.main()
