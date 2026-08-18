from tweetpreprocess.DataDirHelper import DataDirHelper
from nlpvectors.DataframeSplitter import DataframeSplitter

'''
Created on 03.02.2023
This training approach uses a strict 80/20 temporal split: the earliest 80% of tweet groups
are used for training/validation and the latest 20% are used as the test set.
This method reflects a real prediction task where the model is trained on past data and evaluated on future data.

Because the test period is disjoint from the training period, the scores of this script are much
lower than those of the splits that mix periods. The label is the change of the figure of a
reporting period, so it is constant within a quarter and the dataset holds only one label per
quarter. Whenever the same quarter occurs in training and in test, recognizing the period of a text
already determines the label, which is why the mixed splits reach high values. Here that shortcut is
not available and the model has to transfer to quarters it has never seen.

The script therefore prints the composition of the split and the baselines that need no text at all
(majority class, the label of the last training quarter, and the seasonal naive label of the same
calendar quarter of earlier years), so that the score of the model can be judged against them.
Validation uses the latest part of the training period instead of a random sample, because a random
validation set shares its quarters with the training data and would select the checkpoint that
memorizes periods best rather than the one that transfers.
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
from classifier.LSTMNN import MEAN_POOLING
from tweetpreprocess.EqualClassSampler import EqualClassSampler
from classifier.TemporalSplitDiagnostics import getQuartersOfSplits, printSplitComposition, printBaselines, toDatetime
from PredictionModelPath import AMAZON_REVENUE_10_LSTM_BINARY_CLASS,APPLE__EPS_10_LSTM_MULTI_CLASS

torch.set_float32_matmul_precision('medium')

BALANCE_CLASSES = False
# A random validation set shares its quarters with the training data, so the checkpoint it selects is
# the one that recognizes periods best. The last part of the training period is the only validation
# data that resembles the forecasting situation of the test period.
TEMPORAL_VALIDATION = True
VALIDATION_SIZE = 0.3
# With the original 'last' pooling the model does not fit this data at all: its training loss stays
# at ln(num_classes) and it predicts one class for every tweet group. Averaging the LSTM outputs
# instead lets it train, which on Apple EPS moves the test MCC from 0.000 to 0.286.
POOLING = MEAN_POOLING

if __name__ == "__main__":
    predictionModelPath =APPLE__EPS_10_LSTM_MULTI_CLASS

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
    df[postTSPColumn] = toDatetime(df[postTSPColumn])

    # Pre-build index for fast lookups
    tweet_id_to_date = dict(zip(df["tweet_id"], df[postTSPColumn]))
    tweet_id_to_class = dict(zip(df["tweet_id"], df["class"]))

    # Sort splits by earliest tweet date (temporal order)
    split_dates = [min(tweet_id_to_date[tid] for tid in split) for split in tweetSplits]
    sorted_indices = np.argsort(split_dates)
    tweetSplits = [tweetSplits[i] for i in sorted_indices]

    split_labels = np.array([tweet_id_to_class[split[0]] for split in tweetSplits])

    # Strict 80/20 temporal split: first 80% for train+val, last 20% for test
    n = len(tweetSplits)
    split_point = int(n * 0.8)
    train_val_idx = np.arange(0, split_point)
    test_idx = np.arange(split_point, n)

    testIdxPath = predictionModelPath.getModelPath() + "\\test_idx_fold0.npy"
    np.save(testIdxPath, test_idx)

    train_val_labels = split_labels[train_val_idx]
    if TEMPORAL_VALIDATION:
        # Latest part of the training period as validation, so that model selection is driven by
        # transfer to a later period instead of by recognition of periods already seen.
        val_start = int(len(train_val_idx) * (1 - VALIDATION_SIZE))
        train_idx, val_idx = train_val_idx[:val_start], train_val_idx[val_start:]
    else:
        train_idx, val_idx = train_test_split(train_val_idx, random_state=1337, test_size=VALIDATION_SIZE, stratify=train_val_labels)

    split_quarters = getQuartersOfSplits(df, tweetSplits)
    printSplitComposition(split_quarters, train_idx, val_idx, test_idx, split_labels)
    printBaselines(split_quarters, split_labels, train_val_idx, test_idx)

    # Compute class weights from training set (inverse frequency)
    train_labels = split_labels[train_idx]
    label_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        cls_val = predictionModelPath.getPredictionClassMapper().index_to_class(cls_idx)
        count = label_counts.get(cls_val, 0)
        class_weights[cls_idx] = total / (num_classes * max(count, 1))

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

    model = CreateClassifierModel(word_vectors=word_vectors, num_classes=num_classes, class_weights=class_weights,
                                  pooling=POOLING, padTokenIdx=pad_token_idx).createModel()
    bestModelPath = Trainer().train(
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
        checkpointName="tweetpredict_fold0"
    )

    # Classification metrics on test set using best checkpoint. The path comes from the trainer,
    # because ModelCheckpoint writes tweetpredict_fold0-v1.ckpt when tweetpredict_fold0.ckpt is
    # already there, and loadModel loads with strict=False and would silently score the older model.
    print("\n=== Classification Metrics (Temporal 80/20 Split) ===")
    print("Best checkpoint", bestModelPath)
    bestModel = loadModel(bestModelPath, word_vectors, num_classes=num_classes, pooling=POOLING, padTokenIdx=pad_token_idx)
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

    printBaselines(split_quarters, split_labels, train_val_idx, test_idx)

    # Per quarter the label is constant, so the model effectively casts one vote per test quarter.
    # Comparing that vote with the label of the preceding quarter shows whether the model transfers
    # or only repeats the most recent period it was trained on.
    print("\n=== Prediction per test quarter ===")
    predictionsPerQuarter = np.array(prediction_classes)
    testQuarters = split_quarters[test_idx]
    allQuarters = sorted(set(split_quarters.tolist()))
    for quarter in sorted(set(testQuarters.tolist())):
        mask = testQuarters == quarter
        majorityPrediction = Counter(predictionsPerQuarter[mask].tolist()).most_common(1)[0][0]
        previousQuarter = allQuarters[allQuarters.index(quarter) - 1]
        previousLabel = Counter(split_labels[split_quarters == previousQuarter].tolist()).most_common(1)[0][0]
        print("  %s true %s, predicted %s, label of %s was %s%s"
              % (quarter, split_labels[test_idx][mask][0], majorityPrediction, previousQuarter, previousLabel,
                 "   <- repeats the previous quarter" if majorityPrediction == previousLabel else ""))

