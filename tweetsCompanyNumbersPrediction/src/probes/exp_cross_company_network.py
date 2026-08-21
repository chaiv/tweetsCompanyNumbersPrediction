"""Audit the cross-company linguistic clock without shared tweets or shared authors.

This probe distinguishes three possible carriers of cross-company quarter recognition:

1. identical tweet IDs stored for more than one company;
2. authors who post about more than one company;
3. language that transfers even between company-exclusive authors.

The task is quarter dating, not financial forecasting. Groups contain ten consecutive tweets,
the TF-IDF vocabulary is fitted only on the source company, and no financial CSV is loaded.
Run from tweetsCompanyNumbersPrediction/src. CPU only; the full default run takes several minutes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from tweetpreprocess.DataDirHelper import DataDirHelper


FILES = {
    "Apple": "CompanyTweetsAppleWithEpsMulticlass.csv",
    "Amazon": "amazonTweetsWithNumbersMulticlass.csv",
    "Tesla": "CompanyTweetsTeslaWithCarSalesMulticlass.csv",
}
GROUP_SIZE = 10


def load_frames():
    directory = os.path.join(DataDirHelper().getDataDir(), "companyTweets")
    return {
        name: pd.read_csv(
            os.path.join(directory, filename),
            usecols=["tweet_id", "writer", "post_date", "body"],
        )
        for name, filename in FILES.items()
    }


def pairwise_union(values):
    keys = list(values)
    shared = set()
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            shared.update(values[first].intersection(values[second]))
    return shared


def build_groups(frame, excluded_ids, excluded_authors, maximum, random_state):
    selected = frame[
        ~frame["tweet_id"].isin(excluded_ids)
        & ~frame["writer"].isin(excluded_authors)
    ].sort_values("post_date").reset_index(drop=True)
    indexes = np.arange(0, len(selected) - GROUP_SIZE + 1, GROUP_SIZE)
    if len(indexes) > maximum:
        indexes = np.sort(random_state.choice(indexes, maximum, replace=False))
    bodies = selected["body"].fillna("").astype(str).values
    dates = pd.to_datetime(selected["post_date"], unit="s")
    texts = [" ".join(bodies[start:start + GROUP_SIZE]) for start in indexes]
    quarters = dates.iloc[indexes + GROUP_SIZE // 2].dt.to_period("Q").astype(str).values
    return texts, quarters, len(selected)


def classifier():
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-6,
        max_iter=25,
        tol=1e-4,
        random_state=0,
        n_jobs=1,
    )


def transfer_matrix(groups, min_df, max_features):
    names = list(groups)
    results = {}
    for source in names:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        source_features = vectorizer.fit_transform(groups[source][0])
        fitted = classifier().fit(source_features, groups[source][1])
        for target in names:
            if target == source:
                continue
            target_features = vectorizer.transform(groups[target][0])
            results[(source, target)] = accuracy_score(
                groups[target][1], fitted.predict(target_features))
    return results


def print_matrix(title, names, results):
    print("\n" + title)
    print("source \\ target  " + "  ".join("%8s" % name for name in names))
    for source in names:
        values = []
        for target in names:
            values.append("       -" if target == source else "%8.3f" % results[(source, target)])
        print("%-12s  %s" % (source, "  ".join(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--id-exclusive-groups", type=int, default=60000)
    parser.add_argument("--author-exclusive-groups", type=int, default=30000)
    args = parser.parse_args()

    frames = load_frames()
    names = list(frames)
    id_sets = {name: set(frame["tweet_id"].tolist()) for name, frame in frames.items()}
    author_sets = {
        name: set(frame["writer"].dropna().tolist()) for name, frame in frames.items()
    }
    shared_ids = pairwise_union(id_sets)
    shared_authors = pairwise_union(author_sets)

    print("[A] pairwise shared tweet IDs")
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = id_sets[first].intersection(id_sets[second])
            print(
                "  %s/%s: %d unique IDs; shares %.4f / %.4f"
                % (
                    first,
                    second,
                    len(overlap),
                    len(overlap) / len(id_sets[first]),
                    len(overlap) / len(id_sets[second]),
                )
            )

    without_shared_ids = {
        name: frame[~frame["tweet_id"].isin(shared_ids)] for name, frame in frames.items()
    }
    authors_without_shared_ids = {
        name: set(frame["writer"].dropna().tolist())
        for name, frame in without_shared_ids.items()
    }
    all_three_authors = set.intersection(*authors_without_shared_ids.values())
    print("\n[B] common information-broker network after shared-ID removal")
    print("  authors active for all three companies:", len(all_three_authors))
    for name, frame in without_shared_ids.items():
        print(
            "  %s: %.4f of tweets written by all-three authors"
            % (name, frame["writer"].isin(all_three_authors).mean())
        )

    random_state = np.random.RandomState(args.seed)
    id_exclusive = {
        name: build_groups(
            frame, shared_ids, set(), args.id_exclusive_groups, random_state)
        for name, frame in frames.items()
    }
    print(
        "\nID-exclusive groups:",
        {name: len(values[0]) for name, values in id_exclusive.items()},
    )
    id_results = transfer_matrix(id_exclusive, min_df=5, max_features=150000)
    print_matrix("[C] source-only vocabulary, shared tweet IDs removed", names, id_results)

    random_state = np.random.RandomState(args.seed)
    author_exclusive = {
        name: build_groups(
            frame,
            shared_ids,
            shared_authors,
            args.author_exclusive_groups,
            random_state,
        )
        for name, frame in frames.items()
    }
    print("\n[D] company-exclusive authors")
    for name, (texts, quarters, tweet_count) in author_exclusive.items():
        majority = pd.Series(quarters).value_counts(normalize=True).iloc[0]
        print(
            "  %s: %d tweets, %d groups, majority baseline %.3f"
            % (name, tweet_count, len(texts), majority)
        )
    author_results = transfer_matrix(author_exclusive, min_df=3, max_features=100000)
    print_matrix("[E] shared tweet IDs and cross-company authors removed", names, author_results)


if __name__ == "__main__":
    main()
