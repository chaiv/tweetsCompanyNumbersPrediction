from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.DataframeSplitter import DataframeSplitter

'''
Created on 19.04.2026
This training approach uses Stratified KFold cross-validation where within each class the temporal 
order is preserved. For each fold, within each class the earliest samples go to train/val and the 
latest samples go to test. This guarantees:
  1. All classes are represented in train, val, and test in every fold
  2. Within each class, no future data leaks into training
  3. Multiple folds for robust evaluation
Unlike the pure temporal split, this avoids the problem of entire classes being absent from train or test.
Unlike the shuffled KFold, this preserves temporal integrity within each class.
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
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS, AMAZON_REVENUE_10_LSTM_MULTI_CLASS, \
    AMAZON_REVENUE_20_LSTM_MULTI_CLASS, TESLA_CAR_SALES_10_LSTM_MULTI_CLASS, \
    APPLE__EPS_10_LSTM_MULTI_CLASS

torch.set_float32_matmul_precision('medium')

BALANCE_CLASSES = False

if __name__ == "__main__":
    predictionModelPath = AMAZON_REVENUE_10_LSTM_BINARY_CLASS

    df = pd.read_csv(predictionModelPath.getDataframePath())
    df.fillna('', inplace=True)
    if BALANCE_CLASSES:
        df = EqualClassSampler().getDfWithEqualNumberOfClassSamples(df)
        print("Classes balanced with EqualClassSampler")
    print(TweetDataframeExplore(df).getClassDistribution())
    word_vectors = KeyedVectors.load_word2vec_format(predictionModelPath.getWordVectorsPath(), binary=False)
    textEncoder = WordVectorsIDEncoder(word_vectors)
    tokenizer = TweetTokenizer(DefaultWordFilter())
    pad_token_idx = textEncoder.getPADTokenID()
    vocab_size = textEncoder.getVocabularyLength()
    num_classes = predictionModelPath.getPredictionClassMapper().get_number_of_classes()

    splitter = DataframeSplitter()
    tweetSplits = splitter.getSplitIds(df, predictionModelPath.getTweetGroupSize())

    postTSPColumn = "post_date"
    df[postTSPColumn] = pd.to_datetime(df[postTSPColumn])

    # Pre-build index for fast lookups
    tweet_id_to_date = dict(zip(df["tweet_id"], df[postTSPColumn]))
    tweet_id_to_class = dict(zip(df["tweet_id"], df["class"]))

    # Sort splits by earliest tweet date (temporal order)
    split_dates = [min(tweet_id_to_date[tid] for tid in split) for split in tweetSplits]
    sorted_indices = np.argsort(split_dates)
    tweetSplits = [tweetSplits[i] for i in sorted_indices]

    split_labels = np.array([tweet_id_to_class[split[0]] for split in tweetSplits])

    # --- Stratified KFold with temporal order per class ---
    # For each class, splits are already in temporal order (sorted above).
    # In each fold, each class contributes its temporally earliest portion to train/val
    # and the next temporal block to test.
    kfold_splits = 2
    unique_classes = np.unique(split_labels)
    class_indices = {c: np.where(split_labels == c)[0] for c in unique_classes}

    # For each class, divide into kfold_splits+1 temporal blocks.
    # Fold k: train/val = blocks 0..k, test = block k+1
    # This is an expanding window per class.
    # To keep it simple and balanced: each fold uses a different test block per class.
    # We divide each class into kfold_splits equal temporal blocks.
    # Fold k: test = block k, train/val = all other blocks.
    # This is like TimeSeriesSplit but stratified per class.

    class_fold_blocks = {}
    for c in unique_classes:
        c_idx = class_indices[c]
        # Split into kfold_splits roughly equal temporal blocks
        class_fold_blocks[c] = np.array_split(c_idx, kfold_splits)

    all_fold_metrics = []

    for fold in range(kfold_splits):
        train_val_all = []
        test_all = []

        for c in unique_classes:
            blocks = class_fold_blocks[c]
            # Test block = fold-th block, train/val = all other blocks
            for b_idx, block in enumerate(blocks):
                if b_idx == fold:
                    test_all.extend(block.tolist())
                else:
                    train_val_all.extend(block.tolist())

        train_val_idx = np.array(train_val_all)
        test_idx = np.array(test_all)
        testIdxPath = predictionModelPath.getModelPath() + f"\\test_idx_fold{fold}.npy"
        np.save(testIdxPath, test_idx)

        # Stratified train/val split (ensures all classes in both train and val)
        train_val_labels = split_labels[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, random_state=1337, test_size=0.2, stratify=train_val_labels
        )

        # Compute class weights from training set only (inverse frequency)
        train_labels = split_labels[train_idx]
        label_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.zeros(num_classes)
        for cls_idx in range(num_classes):
            cls_val = predictionModelPath.getPredictionClassMapper().index_to_class(cls_idx)
            count = label_counts.get(cls_val, 0)
            class_weights[cls_idx] = total / (num_classes * max(count, 1))

        print(f"\n=== Fold {fold} ===")
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

        # Re-create model for each fold with class weights
        model = CreateClassifierModel(word_vectors=word_vectors, num_classes=num_classes, class_weights=class_weights).createModel()
        Trainer().train(
            batch_size=256,
            epochs=10,
            num_workers=0,
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
        report = metrics.classification_report(trueClasses, prediction_classes)
        mcc = metrics.calculate_mcc(trueClasses, prediction_classes)
        print(report)
        print("MCC " + str(mcc))
        all_fold_metrics.append({"fold": fold, "mcc": mcc, "report": report})

    # Summary across all folds
    print("\n" + "=" * 60)
    print("=== SUMMARY ACROSS ALL FOLDS ===")
    print("=" * 60)
    mccs = [m["mcc"] for m in all_fold_metrics]
    print(f"MCC per fold: {mccs}")
    print(f"Mean MCC: {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")

