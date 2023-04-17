from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import unittest

class TestCoherenceModel(unittest.TestCase):

    def setUp(self):
        # Define example data
        self.documents = [
            ["this", "first", "document"],
            ["that", "second", "text"]
        ]
        # Create dictionary
        self.dictionary = Dictionary(self.documents)

    def test_coherence_model(self):
        # Define example topics
        topics = [
            ["this", "first", "document"],
            ["that", "second", "text"]
        ]
        # Calculate coherence score
        cm = CoherenceModel(topics=topics, texts=self.documents, dictionary=self.dictionary, coherence='c_v')
        score = cm.get_coherence()
        # Assert that coherence score is not None
        print(score)
        self.assertIsNotNone(score)