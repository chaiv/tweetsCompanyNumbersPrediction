'''
Created on 06.03.2023

@author: vital
'''
import unittest
import tempfile
import json
import os
from nlpvectors.VocabularyIDEncoder import VocabularyIDEncoder

class TestVocabularyIDEncoder(unittest.TestCase):

    def setUp(self):
        # Create a temporary file in the same directory as the test
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(json.dumps({"hello": 1, "world": 2}).encode())
        self.temp_file.close()

        # Create the encoder object using the temporary file
        self.encoder = VocabularyIDEncoder(self.temp_file.name)

    def tearDown(self):
        # Delete the temporary file
        os.unlink(self.temp_file.name)

    def test_getMaxWordsAmount(self):
        self.assertEqual(self.encoder.getMaxWordsAmount(), 80)

    def test_getVocabularyLength(self):
        self.assertEqual(self.encoder.getVocabularyLength(), 2)

    def test_getPADTokenID(self):
        self.assertEqual(self.encoder.getPADTokenID(), 3)

    def test_encodeTokens(self):
        tokens = ["hello", "world"]
        expected_output = [1, 2]
        self.assertEqual(self.encoder.encodeTokens(tokens), expected_output)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()