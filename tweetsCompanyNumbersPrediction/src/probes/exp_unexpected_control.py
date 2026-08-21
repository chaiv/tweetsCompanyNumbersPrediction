"""Control: does the linguistic clock survive removing the top 1% of accounts and all exact duplicates?

Probe from the August 2026 review; see surprising-findings.en.md in the repository root.
Run from tweetsCompanyNumbersPrediction/src with that directory on the PYTHONPATH. CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tweetpreprocess.DataDirHelper import DataDirHelper
import numpy as np, pandas as pd, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
D = os.path.join(DataDirHelper().getDataDir(), "companyTweets") + os.sep
FILES = {"Apple": "CompanyTweetsAppleWithEpsMulticlass.csv", "Amazon": "amazonTweetsWithNumbersMulticlass.csv",
         "Tesla": "CompanyTweetsTeslaWithCarSalesMulticlass.csv"}
N = 10; RS = np.random.RandomState(0)

def load_clean(name):
    df = pd.read_csv(D + FILES[name], usecols=["post_date", "body", "writer"])
    df["body"] = df["body"].fillna(""); n0 = len(df)
    vc = df["writer"].value_counts(); top = set(vc.head(max(1, len(vc) // 100)).index)
    df = df[~df["writer"].isin(top)]
    df = df.drop_duplicates("body")
    df["dt"] = pd.to_datetime(df.post_date, unit="s")
    df = df.sort_values("post_date").reset_index(drop=True)
    print(f"  {name}: kept {len(df):,} of {n0:,} tweets after removing top-1% accounts and duplicates")
    return df

def groups(df, n_groups=60000):
    idx = np.arange(0, len(df) - N + 1, N)
    if len(idx) > n_groups: idx = np.sort(RS.choice(idx, n_groups, replace=False))
    b = df["body"].values; d = df["dt"].values
    return [" ".join(b[i:i + N]) for i in idx], pd.DatetimeIndex([d[i + N // 2] for i in idx])

def clf(): return SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=25, tol=1e-4, random_state=0, n_jobs=-1)

print("[control] data after removing automated accounts and duplicates")
G = {k: groups(load_clean(k)) for k in FILES}
vec = TfidfVectorizer(min_df=5, max_features=150000, sublinear_tf=True, dtype=np.float32).fit(sum((G[k][0] for k in G), []))
X = {k: vec.transform(G[k][0]) for k in G}; Q = {k: G[k][1].to_period("Q").astype(str).values for k in G}

print("\n[A'] cross-company quarter dating, cleaned")
for src in G:
    c = clf().fit(X[src], Q[src]); row = []
    for tgt in G:
        if tgt == src:
            perm = RS.permutation(len(Q[src])); cut = int(0.8 * len(perm))
            acc = accuracy_score(Q[src][perm[cut:]], clf().fit(X[src][perm[:cut]], Q[src][perm[:cut]]).predict(X[src][perm[cut:]]))
        else:
            acc = accuracy_score(Q[tgt], c.predict(X[tgt]))
        row.append(f"{tgt}: {acc:.3f}")
    print(f"  trained on {src:6s} -> " + "   ".join(row))

print("\n[B'] dating resolution within Apple, cleaned")
k = "Apple"; perm = RS.permutation(len(Q[k])); cut = int(0.8 * len(perm)); tr, te = perm[:cut], perm[cut:]
for label, fmt in [("quarter", None), ("month", "%Y-%m"), ("ISO week", "%G-W%V")]:
    y = Q[k] if fmt is None else G[k][1].strftime(fmt).values
    p = clf().fit(X[k][tr], y[tr]).predict(X[k][te])
    print(f"  {label:9s}: {len(set(y))} classes, accuracy {accuracy_score(y[te], p):.3f}")

print("\n[F] what carries the clock? top dated features for a few Apple quarters (cleaned data)")
c = clf().fit(X[k], Q[k]); names = np.array(vec.get_feature_names_out())
for q in ["2015Q3", "2016Q4", "2018Q1", "2019Q3"]:
    i = list(c.classes_).index(q); top = names[np.argsort(c.coef_[i])[::-1][:12]]
    print(f"  {q}: {', '.join(top)}")
print("\nFINISHED")
