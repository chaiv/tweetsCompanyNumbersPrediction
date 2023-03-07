'''
Created on 06.03.2023

@author: vital
'''
import unittest


from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter


class TestDefaultWordFilter(unittest.TestCase):
    def setUp(self):
        self.filter = DefaultWordFilter()
        
    def test_convert_to_lowercase(self):
        self.assertEqual(self.filter.convert_to_lowercase("AbC"), "abc")
        
    def test_remove_urls(self):
        self.assertEqual(self.filter.remove_urls("http://example.com"), "")
        
    def test_remove_mentions(self):
        self.assertEqual(self.filter.remove_mentions("@example"), "")
        
    def test_remove_hashtags(self):
        self.assertEqual(self.filter.remove_hashtags("#example"), "")
        
    def test_remove_stopwords(self):
        self.assertEqual(self.filter.remove_stopwords("is"), "")
        
    def test_remove_punctuation(self):
        self.assertEqual(self.filter.remove_punctuation("hello,!"), "hello")
        
    def test_remove_digits(self):
        self.assertEqual(self.filter.remove_digits("123abc456"), "abc")
        
    def test_stem(self):
        self.assertEqual(self.filter.stem("running"), "run")
        self.assertEqual(self.filter.stem("walked"), "walk")
    
    def test_simple_preprocess(self):
        self.assertEqual(self.filter.simple_preprocess("l"), "")
        self.assertEqual(self.filter.simple_preprocess("l2"), "")
        self.assertEqual(self.filter.simple_preprocess("tooooooooooooloooooong"), "")
    
    def testFilter(self):
        print(self.filter.filter("-Check"))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()