'''
Created on 14.11.2023

@author: vital
'''
from exploredata.POSTagging import PartOfSpeechTagging

def addUntokenizedWordColumnFromTweetDf(tweetDf,importantWordsDf,tweetIdColumnName = "tweet_id",tweetColumnName = "body",tokenIndexColumnName = "token_index",bodyWordsColumnName = "body_words",originalTokenColumnName = "original_token"):
    merged_df = importantWordsDf.merge(tweetDf, on=tweetIdColumnName)
    merged_df[bodyWordsColumnName] = merged_df[tweetColumnName].str.split()
    merged_df[originalTokenColumnName] = merged_df.apply(
        lambda row: row[bodyWordsColumnName][row[tokenIndexColumnName]] if 0 <= row[tokenIndexColumnName] < len(row[bodyWordsColumnName]) else None,axis=1)
    return merged_df


def addPOSTagsColumn(posTagging : PartOfSpeechTagging, importantWordsDf,tweetColumnName = "body",tokenIndexColumnName = "token_index",tweetPosColumnName = 'tweet_pos',tokenPosColumnName = 'token_pos'):
    importantWordsDf[tweetPosColumnName] = importantWordsDf.apply(
        lambda row:  posTagging.getPOSTagsAsStrList(row[tweetColumnName]) ,axis=1)
    importantWordsDf[tokenPosColumnName] = importantWordsDf.apply(
        lambda row:  row[tweetPosColumnName][row[tokenIndexColumnName]] ,axis=1)
    return importantWordsDf

def addTopicColumns(topicExtractor, importantWordsDf, tweetIdColumnName='tweet_id', topicNumColumnName='topic_num', topicWordsColumnName='topic_words'):
    doc_topics, doc_dist, topic_words, topic_word_scores = topicExtractor.get_documents_topics(importantWordsDf[tweetIdColumnName].tolist())
    importantWordsDf[topicNumColumnName] = doc_topics.tolist()
    importantWordsDf[topicWordsColumnName] = topic_words.tolist()
    return importantWordsDf

def addTopicOriginalWordsColumn(tokenizerLookupDict,importantWordsDf,topicWordsColumnName='topic_words',originalTopicWordsColumnName = 'original_topic_words'):
    importantWordsDf[originalTopicWordsColumnName] = importantWordsDf.apply(
        lambda row:  [(tokenizerLookupDict[topicWord] if topicWord in tokenizerLookupDict else '') for topicWord in row[topicWordsColumnName]],axis=1)
    return importantWordsDf
    
        