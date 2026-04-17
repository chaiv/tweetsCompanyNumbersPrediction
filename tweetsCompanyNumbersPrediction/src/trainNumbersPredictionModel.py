from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.DataframeSplitter import DataframeSplitter

'''
Created on 03.02.2023

@author: vital
'''
import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from nlpvectors.TweetTokenizer import TweetTokenizer
from tweetpreprocess.wordfiltering.DefaultWordFilter import DefaultWordFilter
from exploredata.TweetDataframeExplore import TweetDataframeExplore
from nlpvectors.WordVectorsIDEncoder import WordVectorsIDEncoder
from classifier.Trainer import Trainer
from classifier.TweetGroupDataset import TweetGroupDataset

from classifier.CreateClassifierModel import CreateClassifierModel
from classifier.transformer.Predictor import Predictor
from classifier.ClassificationMetrics import ClassificationMetrics
from classifier.ModelEvaluationHelper import loadModel
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_MULTI_CLASS,\
    AMAZON_REVENUE_20_LSTM_MULTI_CLASS, TESLA_CAR_SALES_10_LSTM_MULTI_CLASS, \
    APPLE__EPS_10_LSTM_MULTI_CLASS


torch.set_float32_matmul_precision('medium') #needed for quicker cuda 

if  __name__ == "__main__":
    predictionModelPath =  APPLE__EPS_10_LSTM_MULTI_CLASS

    # Read raw dataframe WITHOUT EqualClassSampler (no pre-split balancing)
    df = pd.read_csv(predictionModelPath.getDataframePath())
    df.fillna('', inplace=True) #nan values in body columns 
    print(TweetDataframeExplore(df).getClassDistribution())
    word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()
    
    splitter = DataframeSplitter()
    tweetSplits = splitter.getSplitIds(df, predictionModelPath.getTweetGroupSize())

    # Sort splits chronologically
    postTSPColumn = "post_date"
    df[postTSPColumn] = pd.to_datetime(df[postTSPColumn])

    # Pre-build index for fast lookups (avoid repeated isin() on large dataframes)
    tweet_id_to_date = dict(zip(df["tweet_id"], df[postTSPColumn]))
    tweet_id_to_class = dict(zip(df["tweet_id"], df["class"]))

    split_dates = [min(tweet_id_to_date[tid] for tid in split) for split in tweetSplits]
    sorted_indices = np.argsort(split_dates)
    tweetSplits = [tweetSplits[i] for i in sorted_indices]

    # Get class label for each sorted split
    split_labels = np.array([tweet_id_to_class[split[0]] for split in tweetSplits])

    # Stratified temporal cross-validation:
    # Per class, preserve temporal order. First 70% of each class = train/val, last 30% = test.
    # This ensures all classes are represented in train, val, and test.
    n = len(tweetSplits)
    kfold_splits = 2
    num_classes = predictionModelPath.getPredictionClassMapper().get_number_of_classes()

    # Group indices by class, preserving temporal order within each class
    unique_classes = np.unique(split_labels)
    class_indices = {c: np.where(split_labels == c)[0] for c in unique_classes}

    # Calculate fold steps per class
    class_test_sizes = {c: int(len(idx) * 0.3) for c, idx in class_indices.items()}
    class_remaining = {c: len(idx) - class_test_sizes[c] for c, idx in class_indices.items()}
    class_fold_steps = {c: class_remaining[c] // kfold_splits for c in unique_classes}

    for fold in range(kfold_splits):
        train_val_all = []
        test_all = []
        for c in unique_classes:
            c_indices = class_indices[c]
            c_n = len(c_indices)
            # Expanding window per class
            c_train_val_end = class_test_sizes[c] + class_fold_steps[c] * (fold + 1)
            if fold == kfold_splits - 1:
                c_train_val_end = c_n
            c_train_val_count = int(c_train_val_end * 0.7)
            # First 70% of window = train/val, last 30% = test (temporally ordered per class)
            train_val_all.extend(c_indices[:c_train_val_count].tolist())
            test_all.extend(c_indices[c_train_val_count:c_train_val_end].tolist())

        train_val_idx = np.array(train_val_all)
        test_idx = np.array(test_all)
        testIdxPath = predictionModelPath.getModelPath()+f"\\test_idx_fold{fold}.npy"
        np.save(testIdxPath, test_idx)

        # Stratified train/val split (ensures all classes in both train and val)
        train_val_labels = split_labels[train_val_idx]
        train_idx, val_idx = train_test_split(train_val_idx, random_state=1337, test_size=0.3, stratify=train_val_labels)

        # Compute class weights from training set only (inverse frequency)
        train_labels = split_labels[train_idx]
        label_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            cls_val = predictionModelPath.getPredictionClassMapper().index_to_class(cls_idx)
            count = label_counts.get(cls_val, 0)
            class_weights[cls_idx] = total / (num_classes * max(count, 1))

        print(f"=== Fold {fold} ===")
        print(f"Class weights: {class_weights}")
        print("Train classes", splitter.getClassCountsOfSplitsByIndexes(df, tweetSplits, train_idx))
        print("Val classes", splitter.getClassCountsOfSplitsByIndexes(df, tweetSplits, val_idx))
        print("Test classes", splitter.getClassCountsOfSplitsByIndexes(df, tweetSplits, test_idx))
        train_data = TweetGroupDataset(dataframe=df, splits=tweetSplits, splitIndexes=train_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        val_data = TweetGroupDataset(dataframe=df, splits=tweetSplits, splitIndexes=val_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        test_data = TweetGroupDataset(dataframe=df, splits=tweetSplits, splitIndexes=test_idx, tokenizer=tokenizer, textEncoder=textEncoder)
        print("len(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))

        # Re-create model for each fold with class weights for training loss
        model = CreateClassifierModel(word_vectors=word_vectors, num_classes=num_classes, class_weights=class_weights).createModel()
        Trainer().train(
            batch_size=256,
            epochs=10,
            num_workers=2,
            pad_token_idx=pad_token_idx,
            model=model, 
            train_data=train_data, 
            val_data=val_data, 
            test_data=test_data, 
            loggerPath=DataDirHelper().getDataDir() + 'companyTweets\\modellogs', 
            loggerName="tweetpredict", 
            checkpointPath=predictionModelPath.getModelPath(), 
            checkpointName=f"tweetpredict_fold{fold}"
            )

        # Classification metrics on test set using best checkpoint
        print(f"\n=== Fold {fold} Classification Metrics ===")
        bestModelPath = predictionModelPath.getModelPath() + f"\\tweetpredict_fold{fold}.ckpt"
        bestModel = loadModel(bestModelPath, word_vectors, num_classes=num_classes)
        predictionClassMapper = predictionModelPath.getPredictionClassMapper()

        tweetGroups = []
        trueClasses = []
        for i in range(len(test_data)):
            tweetGroup = test_data.getAsTweetGroup(i)
            tweetGroups.append(tweetGroup)
            trueClasses.append(tweetGroup.getLabel())

        predictor = Predictor(bestModel, tokenizer, textEncoder, predictionClassMapper, None)
        prediction_classes = predictor.predictMultipleAsTweetGroupsInChunks(tweetGroups, 1000)
        print("true_classes counts ", ', '.join(f"{item}: {count}" for item, count in Counter(trueClasses).items()))
        print("prediction_classes counts ", ', '.join(f"{item}: {count}" for item, count in Counter(prediction_classes).items()))
        metrics = ClassificationMetrics()
        print(metrics.classification_report(trueClasses, prediction_classes))
        print("MCC " + str(metrics.calculate_mcc(trueClasses, prediction_classes)))
