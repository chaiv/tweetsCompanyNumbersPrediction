'''
Created on 19.02.2023

@author: vital
'''
import unittest
from tagging.PosDepTagger import PosDepTagger


class TestPosDepTagger(unittest.TestCase):
    
    def setUp(self):
        self.tagger = PosDepTagger()
        
    def test_tagging(self):
        sentence = "The quick brown fox jumps over the lazy dog"
        expected_tags = [
            (0,'The', 'DET', 'det'), 
            (1,'quick', 'ADJ', 'amod'), 
            (2,'brown', 'ADJ', 'amod'), 
            (3,'fox', 'NOUN', 'nsubj'), 
            (4,'jumps', 'VERB', 'ROOT'), 
            (5,'over', 'ADP', 'prep'), 
            (6,'the', 'DET', 'det'), 
            (7,'lazy', 'ADJ', 'amod'), 
            (8,'dog', 'NOUN', 'pobj')
        ]
        
        actual_tags = self.tagger.get_tags(sentence)
        
        self.assertEqual(expected_tags, actual_tags)
        