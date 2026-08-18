"""Measure how accurately tweet-group text identifies its own reporting quarter.

This is a diagnostic, not a financial forecast. A high score demonstrates why evaluations that
place groups from the same quarter in train and test can be solved through period recognition.
The TF-IDF vocabulary is fitted on the training groups only.
"""

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from PredictionModelPath import APPLE__EPS_10_LSTM_MULTI_CLASS
from classifier.QuarterAlignedDataset import build_quarter_groups, select_balanced_quarter_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-per-quarter", type=int, default=512)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    prediction_path = APPLE__EPS_10_LSTM_MULTI_CLASS
    frame = pd.read_csv(
        prediction_path.getDataframePath(),
        usecols=["tweet_id", "post_date", "body", "class"],
    )
    frame["body"] = frame["body"].fillna("")
    frame, groups = build_quarter_groups(frame, prediction_path.getTweetGroupSize())
    # The source corpus ends at 2019-12-31 UTC. In local reporting time its final hour already
    # belongs to 2020Q1, but it is only a tiny partial quarter and is excluded here.
    complete_quarters = sorted({group.quarter for group in groups if int(group.quarter[:4]) <= 2019})
    selected = select_balanced_quarter_groups(
        groups, complete_quarters, args.groups_per_quarter, seed=args.seed)
    texts = [
        " <SEP> ".join(frame.loc[list(group.row_indexes), "body"].astype(str).tolist())
        for group in selected
    ]
    labels = [group.quarter for group in selected]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    classifier = LogisticRegression(
        C=4.0,
        max_iter=300,
        n_jobs=1,
        random_state=args.seed,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    print("Quarter-recognition accuracy: %.4f" % accuracy_score(test_labels, predictions))
    print("Chance level: %.4f" % (1.0 / len(complete_quarters)))
    print("Quarters: %d; train groups: %d; test groups: %d; vocabulary: %d"
          % (len(complete_quarters), len(train_labels), len(test_labels), len(vectorizer.vocabulary_)))


if __name__ == "__main__":
    main()
