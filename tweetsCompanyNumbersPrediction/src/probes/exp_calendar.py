"""Is the text model's out-of-period ceiling literally 'reading the calendar'?
Walk-forward own-quarter prediction under three vocabularies x regularisation strengths:
  full         - everything
  no-calendar  - month names/abbreviations, q1..q4, weekday names and pure numbers removed
  calendar-only- ONLY those tokens
Compared with the seasonal naive rule on the same test quarters.

Probe from the August 2026 review; see surprising-findings.en.md in the repository root.
Run from tweetsCompanyNumbersPrediction/src with that directory on the PYTHONPATH. CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tweetpreprocess.DataDirHelper import DataDirHelper
import re, numpy as np, pandas as pd, warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
warnings.filterwarnings("ignore")
D = os.path.join(DataDirHelper().getDataDir(), "companyTweets") + os.sep
FILES = {"Apple": "CompanyTweetsAppleWithEpsMulticlass.csv", "Amazon": "amazonTweetsWithNumbersMulticlass.csv"}
N = 10
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december",
          "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec"]
DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","mon","tue","wed","thu","fri","sat","sun"]
CAL = set(MONTHS + DAYS + ["q1","q2","q3","q4","1q","2q","3q","4q","quarter","fy","fiscal","holiday","holidays","christmas","xmas",
                           "thanksgiving","blackfriday","cybermonday","primeday","easter","summer","winter","spring","fall","autumn",
                           "2015","2016","2017","2018","2019","2020"])
def is_cal(tok): return tok in CAL or tok.isdigit()

def build(name):
    df = pd.read_csv(D + FILES[name], usecols=["post_date", "body", "class"])
    df["body"] = df["body"].fillna(""); df = df.sort_values("post_date").reset_index(drop=True)
    q = pd.to_datetime(df.post_date, unit="s").dt.to_period("Q")
    qlab = df.groupby(q)["class"].agg(lambda s: int(Counter(s).most_common(1)[0][0]))
    idx = np.arange(0, len(df) - N + 1, N); b = df["body"].values; qv = q.values
    texts = [" ".join(b[i:i + N]) for i in idx]; quarter = qv[idx + N // 2]
    return texts, np.array([str(x) for x in quarter]), [str(x) for x in qlab.index], np.array([qlab[x] for x in quarter])

def run(X, qstr, uq, y, alpha, min_train=8):
    yt, yp, ys = [], [], []
    for i, q in enumerate(uq):
        if i < min_train: continue
        trm = np.isin(qstr, uq[:i]); tem = qstr == q
        c = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=1e-4, class_weight="balanced", random_state=0, n_jobs=-1)
        p = c.fit(X[trm], y[trm]).predict(X[tem])
        cal = int(q[-1]); same = [u for u in uq[:i] if int(u[-1]) == cal]
        s = Counter(int(y[qstr == u][0]) for u in same).most_common(1)[0][0]
        yt.append(y[tem]); yp.append(p); ys.append(np.full(tem.sum(), s))
    yt, yp, ys = map(np.concatenate, (yt, yp, ys))
    return accuracy_score(yt, yp), matthews_corrcoef(yt, yp), accuracy_score(yt, ys), matthews_corrcoef(yt, ys)

for name in FILES:
    texts, qstr, uq, y = build(name)
    print(f"\n{'='*78}\n{name}: {len(texts):,} groups")
    full = TfidfVectorizer(min_df=5, max_features=150000, sublinear_tf=True, dtype=np.float32).fit(texts)
    vocab = np.array(full.get_feature_names_out())
    cal_mask = np.array([is_cal(t) for t in vocab])
    print(f"  vocabulary {len(vocab):,}; calendar tokens {cal_mask.sum():,}")
    Xf = full.transform(texts)
    Xnc = Xf[:, np.where(~cal_mask)[0]]
    Xco = Xf[:, np.where(cal_mask)[0]]
    print(f"  {'vocabulary':14s} {'alpha':>7s}   text acc / mcc      seasonal acc / mcc (same test quarters)")
    for label, X in [("full", Xf), ("no-calendar", Xnc), ("calendar-only", Xco)]:
        for alpha in [1e-6, 1e-4, 1e-3, 1e-2]:
            a, m, sa, sm = run(X, qstr, uq, y, alpha)
            print(f"  {label:14s} {alpha:7.0e}   {a:.3f} / {m:+.3f}        {sa:.3f} / {sm:+.3f}")
print("\nFINISHED")
