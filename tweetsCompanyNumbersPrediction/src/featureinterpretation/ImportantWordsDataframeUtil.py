'''
Created on 14.11.2023

@author: vital
'''
from exploredata.POSTagging import PartOfSpeechTagging

def addUntokenizedWordColumnFromTweetDf(tweetDf,importantWordsDf,tweetIdColumnName = "tweet_id",tweetColumnName = "body",tokenIndexColumnName = "token_index"):
    merged_df = importantWordsDf.merge(tweetDf, on=tweetIdColumnName)
    merged_df['body_words'] = merged_df[tweetColumnName].str.split()
    merged_df['original_token'] = merged_df.apply(
        lambda row: row['body_words'][row[tokenIndexColumnName]] if 0 <= row[tokenIndexColumnName] < len(row['body_words']) else None,axis=1)
    return merged_df


def addPOSTagsColumn(posTagging : PartOfSpeechTagging, importantWordsDf,tweetColumnName = "body",tokenIndexColumnName = "token_index"):
    importantWordsDf['tweet_pos'] = importantWordsDf.apply(
        lambda row:  posTagging.getPOSTagsAsStrList(row[tweetColumnName]) ,axis=1)
    importantWordsDf['token_pos'] = importantWordsDf.apply(
        lambda row:  row['tweet_pos'][row[tokenIndexColumnName]] ,axis=1)
    return importantWordsDf

    