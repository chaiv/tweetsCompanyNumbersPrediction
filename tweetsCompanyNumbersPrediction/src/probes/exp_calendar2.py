"""(1) month/season words ONLY (no digits, no years) as vocabulary; how many groups contain one?
(2) the persistence-echo mechanism: agreement of full-vocabulary predictions with the PREVIOUS
    quarter's label, and the MCC a pure persistence forecaster would get on these label sequences.

Probe from the August 2026 review; see surprising-findings.en.md in the repository root.
Run from tweetsCompanyNumbersPrediction/src with that directory on the PYTHONPATH. CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tweetpreprocess.DataDirHelper import DataDirHelper
import numpy as np, pandas as pd, warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
warnings.filterwarnings("ignore")
D = os.path.join(DataDirHelper().getDataDir(), "companyTweets") + os.sep
FILES = {"Apple": "CompanyTweetsAppleWithEpsMulticlass.csv", "Amazon": "amazonTweetsWithNumbersMulticlass.csv"}
N = 10
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december",
          "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec","q1","q2","q3","q4","holiday","holidays",
          "christmas","xmas","thanksgiving","blackfriday","cybermonday","primeday","easter","summer","winter","spring","autumn","fall"]

def build(name):
    df = pd.read_csv(D + FILES[name], usecols=["post_date", "body", "class"])
    df["body"] = df["body"].fillna(""); df = df.sort_values("post_date").reset_index(drop=True)
    q = pd.to_datetime(df.post_date, unit="s").dt.to_period("Q")
    qlab = df.groupby(q)["class"].agg(lambda s: int(Counter(s).most_common(1)[0][0]))
    idx = np.arange(0, len(df) - N + 1, N); b = df["body"].values; qv = q.values
    texts = [" ".join(b[i:i + N]) for i in idx]; quarter = qv[idx + N // 2]
    return texts, np.array([str(x) for x in quarter]), [str(x) for x in qlab.index], np.array([qlab[x] for x in quarter]), qlab

def wf(X, qstr, uq, y, alpha=1e-4, min_train=8):
    out = {}
    for i, q in enumerate(uq):
        if i < min_train: continue
        trm = np.isin(qstr, uq[:i]); tem = qstr == q
        c = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=1e-4, class_weight="balanced", random_state=0, n_jobs=-1)
        out[q] = (tem, c.fit(X[trm], y[trm]).predict(X[tem]))
    return out

for name in FILES:
    texts, qstr, uq, y, qlab = build(name)
    print(f"\n{'='*78}\n{name}")
    # (1) month-only vocabulary
    cv = CountVectorizer(vocabulary=MONTHS, binary=True); Xm = cv.fit_transform(texts)
    has = np.asarray(Xm.sum(axis=1)).ravel() > 0
    print(f"  groups containing at least one month/season word: {has.mean()*100:.1f}%")
    res = wf(Xm.astype(np.float32), qstr, uq, y)
    yt = np.concatenate([y[m] for m, _ in res.values()]); yp = np.concatenate([p for _, p in res.values()])
    print(f"  month-words-only model, walk-forward: acc {accuracy_score(yt, yp):.3f} mcc {matthews_corrcoef(yt, yp):+.3f}")
    hm = np.concatenate([has[m] for m, _ in res.values()])
    print(f"     on groups WITH a month word: acc {accuracy_score(yt[hm], yp[hm]):.3f} mcc {matthews_corrcoef(yt[hm], yp[hm]):+.3f}"
          f"   | without: acc {accuracy_score(yt[~hm], yp[~hm]):.3f} mcc {matthews_corrcoef(yt[~hm], yp[~hm]):+.3f}")
    # (2) persistence echo
    full = TfidfVectorizer(min_df=5, max_features=150000, sublinear_tf=True, dtype=np.float32).fit_transform(texts)
    res = wf(full, qstr, uq, y, alpha=1e-6)
    prev = {q: int(qlab[list(qlab.index)[i - 1]]) for i, q in enumerate(uq) if i >= 1}
    yp = np.concatenate([p for _, p in res.values()]); yt = np.concatenate([y[m] for m, _ in res.values()])
    ypr = np.concatenate([np.full(m.sum(), prev[q]) for q, (m, _) in res.items()])
    print(f"  full-vocabulary model: agreement of its predictions with the PREVIOUS quarter's label {np.mean(yp == ypr)*100:.1f}%,"
          f" with the TRUE label {np.mean(yp == yt)*100:.1f}%")
    print(f"  pure persistence forecaster on the same quarters: acc {accuracy_score(yt, ypr):.3f} mcc {matthews_corrcoef(yt, ypr):+.3f}")
    seq = [int(v) for v in qlab.values]
    print(f"  label sequence: {seq}  -> adjacent quarters share a label in {np.mean([a == b for a, b in zip(seq, seq[1:])])*100:.0f}% of cases")
print("\nFINISHED")
