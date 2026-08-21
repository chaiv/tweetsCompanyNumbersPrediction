"""Deeper probes.
[E1] Lagged-label test: text of quarter Q -> change of Q-1 (announced DURING Q), walk-forward,
     versus own-quarter nowcast and Q+1 forecast. Separates 'method fails' from 'information absent'.
[E2] Information accrual: out-of-period accuracy by week-of-quarter, for the own label and the
     announced (lagged) label. Announcements: Tesla deliveries ~day 2, Apple/Amazon earnings ~day 25-30.
[E3] Language drift curve: cosine similarity of quarterly TF-IDF centroids vs lag; seasonal recurrence.
[E4] Stable vocabulary the lagged-label model relies on.

Probe from the August 2026 review; see surprising-findings.en.md in the repository root.
Run from tweetsCompanyNumbersPrediction/src with that directory on the PYTHONPATH. CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tweetpreprocess.DataDirHelper import DataDirHelper
import numpy as np, pandas as pd, warnings
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")

D = os.path.join(DataDirHelper().getDataDir(), "companyTweets") + os.sep
FILES = {"Apple": "CompanyTweetsAppleWithEpsMulticlass.csv", "Amazon": "amazonTweetsWithNumbersMulticlass.csv",
         "Tesla": "CompanyTweetsTeslaWithCarSalesMulticlass.csv"}
N = 10

def clf():
    return SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=25, tol=1e-4, class_weight="balanced", random_state=0, n_jobs=-1)

def build(name):
    df = pd.read_csv(D + FILES[name], usecols=["post_date", "body", "class"])
    df["body"] = df["body"].fillna(""); df = df.sort_values("post_date").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df.post_date, unit="s"); df["q"] = df.dt.dt.to_period("Q")
    qlab = df.groupby("q")["class"].agg(lambda s: int(Counter(s).most_common(1)[0][0]))
    qs = list(qlab.index)
    idx = np.arange(0, len(df) - N + 1, N)
    b = df["body"].values; d = df["dt"].values; qv = df["q"].values
    texts = [" ".join(b[i:i + N]) for i in idx]
    mid = idx + N // 2
    quarter = qv[mid]; dts = pd.DatetimeIndex(d[mid])
    day_of_q = (dts - pd.DatetimeIndex([q.start_time for q in quarter])).days.values
    pos = {q: i for i, q in enumerate(qs)}
    own = np.array([qlab[q] for q in quarter])
    lag = np.array([qlab[qs[pos[q] - 1]] if pos[q] >= 1 else -1 for q in quarter])
    lead = np.array([qlab[qs[pos[q] + 1]] if pos[q] + 1 < len(qs) else -1 for q in quarter])
    qstr = np.array([str(q) for q in quarter])
    return texts, qstr, [str(q) for q in qs], own, lag, lead, day_of_q

def walk_forward(X, qstr, uq, y, min_train=8):
    out = {}
    for i, q in enumerate(uq):
        if i < min_train: continue
        trm = np.isin(qstr, uq[:i]) & (y >= 0); tem = (qstr == q) & (y >= 0)
        if tem.sum() == 0 or len(set(y[trm])) < 2: continue
        out[q] = (tem, clf().fit(X[trm], y[trm]).predict(X[tem]))
    return out

def pooled(res, y):
    yt = np.concatenate([y[m] for m, _ in res.values()]); yp = np.concatenate([p for _, p in res.values()])
    return accuracy_score(yt, yp), matthews_corrcoef(yt, yp), yt, yp

for name in FILES:
    print(f"\n{'='*78}\n{name}")
    texts, qstr, uq, own, lag, lead, doq = build(name)
    vec = TfidfVectorizer(min_df=5, max_features=150000, sublinear_tf=True, dtype=np.float32)
    X = vec.fit_transform(texts)
    print(f"  {len(texts):,} label-free groups, quarters {uq[0]}..{uq[-1]}")

    print("  [E1] walk-forward, out-of-period, same test quarters for all three targets")
    results = {}
    for tag, y in [("own quarter  (nowcast, the thesis task)", own),
                   ("previous quarter (announced during this quarter)", lag),
                   ("next quarter (true forecast)", lead)]:
        res = walk_forward(X, qstr, uq, y)
        acc, mcc, yt, yp = pooled(res, y)
        quarters_right = sum(Counter(p).most_common(1)[0][0] == y[m][0] for m, p in res.values())
        results[tag] = (res, y)
        print(f"     {tag:50s} acc {acc:.3f}  mcc {mcc:+.3f}  quarters right {quarters_right}/{len(res)}")

    print("  [E2] accuracy by week of quarter (pooled over test quarters)")
    for tag in ["own quarter  (nowcast, the thesis task)", "previous quarter (announced during this quarter)"]:
        res, y = results[tag]; byweek = defaultdict(lambda: [0, 0])
        for m, p in res.values():
            w = np.minimum(doq[m] // 7 + 1, 13)
            for wk, ok in zip(w, p == y[m]):
                byweek[wk][0] += ok; byweek[wk][1] += 1
        line = " ".join(f"w{k}:{byweek[k][0]/byweek[k][1]:.2f}" for k in sorted(byweek))
        print(f"     {tag[:16]:16s} {line}")

    print("  [E3] language drift: centroid cosine similarity by lag (quarters)")
    cent = np.vstack([np.asarray(X[qstr == q].mean(axis=0)) for q in uq])
    S = cosine_similarity(cent); n = len(uq)
    bylag = {l: np.mean([S[i, i + l] for i in range(n - l)]) for l in range(1, 9)}
    print("     " + "  ".join(f"lag{l}:{v:.3f}" for l, v in bylag.items()))
    same_cal = [S[i, i + 4] for i in range(n - 4)]; adj = [S[i, i + 1] for i in range(n - 1)]
    lag3 = [S[i, i + 3] for i in range(n - 3)]; lag5 = [S[i, i + 5] for i in range(n - 5)]
    print(f"     same calendar quarter one year later (lag 4): {np.mean(same_cal):.3f}   vs lag 3: {np.mean(lag3):.3f}  lag 5: {np.mean(lag5):.3f}"
          f"   -> seasonal language recurrence {'YES' if np.mean(same_cal) > max(np.mean(lag3), np.mean(lag5)) else 'no'}")

    print("  [E4] vocabulary of the previous-quarter (announced) model, trained on all quarters")
    y = lag; m = y >= 0; c = clf().fit(X[m], y[m]); names = np.array(vec.get_feature_names_out())
    for ci, cls in enumerate(c.classes_):
        top = names[np.argsort(c.coef_[ci])[::-1][:10]]
        print(f"     class {cls}: {', '.join(top)}")
print("\nFINISHED")
