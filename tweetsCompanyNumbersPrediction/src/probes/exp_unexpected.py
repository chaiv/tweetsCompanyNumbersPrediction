"""Probes for unexpected structure in the data, beyond the audit.
[A] cross-company dating  [B] dating resolution  [C] authorship / bot structure
[D] Q+1 forecast vs calendar  [E] tweet volume vs metric change

Probe from the August 2026 review; see surprising-findings.en.md in the repository root.
Run from tweetsCompanyNumbersPrediction/src with that directory on the PYTHONPATH. CPU only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tweetpreprocess.DataDirHelper import DataDirHelper
import numpy as np, pandas as pd, warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

D = os.path.join(DataDirHelper().getDataDir(), "companyTweets") + os.sep
FILES = {"Apple": "CompanyTweetsAppleWithEpsMulticlass.csv",
         "Amazon": "amazonTweetsWithNumbersMulticlass.csv",
         "Tesla": "CompanyTweetsTeslaWithCarSalesMulticlass.csv"}
N = 10
RS = np.random.RandomState(0)

def load(name):
    df = pd.read_csv(D + FILES[name], usecols=["post_date", "body", "writer", "class"])
    df["body"] = df["body"].fillna("")
    df["dt"] = pd.to_datetime(df.post_date, unit="s")
    df = df.sort_values("post_date").reset_index(drop=True)
    return df

def time_groups(df, n_groups=None):
    """label-free groups of N consecutive tweets -> text, mid date, quarter, month, week, class"""
    idx = np.arange(0, len(df) - N + 1, N)
    if n_groups and len(idx) > n_groups:
        idx = np.sort(RS.choice(idx, n_groups, replace=False))
    texts, dates, cls = [], [], []
    bodies = df["body"].values; dts = df["dt"].values; classes = df["class"].values
    for i in idx:
        texts.append(" ".join(bodies[i:i + N]))
        dates.append(dts[i + N // 2])
        cls.append(int(classes[i]))
    dates = pd.DatetimeIndex(dates)
    return texts, dates, np.array(cls)

def clf():
    return SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=25, tol=1e-4, random_state=0, n_jobs=-1)

data = {k: load(k) for k in FILES}
G = {k: time_groups(data[k], 60000) for k in FILES}
print({k: len(G[k][0]) for k in G}, "groups sampled per company")

# ---------------- [A] cross-company dating ----------------
print("\n[A] CROSS-COMPANY QUARTER DATING (train on one company's tweets, date another's)")
vec = TfidfVectorizer(min_df=5, max_features=150000, sublinear_tf=True, dtype=np.float32)
all_text = sum((G[k][0] for k in G), [])
vec.fit(all_text)                     # shared vocabulary across companies
X = {k: vec.transform(G[k][0]) for k in G}
Q = {k: G[k][1].to_period("Q").astype(str).values for k in G}
for src in G:
    c = clf().fit(X[src], Q[src])
    row = []
    for tgt in G:
        if tgt == src:
            # within-company: random 80/20
            perm = RS.permutation(len(Q[src])); cut = int(0.8 * len(perm))
            c2 = clf().fit(X[src][perm[:cut]], Q[src][perm[:cut]])
            acc = accuracy_score(Q[src][perm[cut:]], c2.predict(X[src][perm[cut:]]))
        else:
            acc = accuracy_score(Q[tgt], c.predict(X[tgt]))
        row.append(f"{tgt}: {acc:.3f}")
    print(f"  trained on {src:6s} -> " + "   ".join(row) + "   (chance 0.05)")
# how far off are cross-company errors? median error in quarters
src, tgt = "Apple", "Amazon"
c = clf().fit(X[src], Q[src]); p = c.predict(X[tgt])
toq = lambda s: int(s[:4]) * 4 + int(s[-1])
err = np.abs(np.array([toq(a) for a in p]) - np.array([toq(a) for a in Q[tgt]]))
print(f"  Apple->Amazon: exact {np.mean(err==0):.3f}, within +-1 quarter {np.mean(err<=1):.3f}, median error {np.median(err):.0f} quarters")

# ---------------- [B] dating resolution ----------------
print("\n[B] DATING RESOLUTION within Apple (random 80/20 on label-free 10-tweet groups)")
k = "Apple"; perm = RS.permutation(len(Q[k])); cut = int(0.8 * len(perm)); tr, te = perm[:cut], perm[cut:]
dates = G[k][1]
for label, fmt in [("year", "%Y"), ("quarter", None), ("month", "%Y-%m"), ("ISO week", "%G-W%V")]:
    y = dates.to_period("Q").astype(str).values if fmt is None else dates.strftime(fmt).values
    c = clf().fit(X[k][tr], y[tr]); p = c.predict(X[k][te])
    print(f"  {label:9s}: {len(set(y))} classes, accuracy {accuracy_score(y[te], p):.3f} (chance {1/len(set(y)):.3f})")
# continuous: regress day index, report median absolute error in days
from sklearn.linear_model import Ridge
dayidx = (dates - dates.min()).days.values.astype(float)
r = Ridge(alpha=1.0).fit(X[k][tr], dayidx[tr]); pd_ = r.predict(X[k][te])
print(f"  continuous date regression: median abs error {np.median(np.abs(pd_ - dayidx[te])):.0f} days over a {dayidx.max():.0f}-day span")

# ---------------- [C] authorship structure ----------------
print("\n[C] WHO WRITES THESE TWEETS?")
for k in data:
    df = data[k]; vc = df["writer"].value_counts(); n = len(df)
    top10 = vc.head(10).sum() / n; top1pct = vc.head(max(1, len(vc) // 100)).sum() / n
    dup_mask = df.duplicated("body", keep=False)
    dup_writers = df.loc[dup_mask, "writer"].value_counts()
    print(f"  {k:6s}: {len(vc):,} authors; top-10 accounts write {top10*100:.1f}% of tweets, top 1% of accounts write {top1pct*100:.1f}%;"
          f" {dup_mask.mean()*100:.1f}% of tweets have an exact twin, top-10 duplicators own {dup_writers.head(10).sum()/max(dup_mask.sum(),1)*100:.1f}% of those")
    print(f"          top accounts: {', '.join(f'{w} ({c:,})' for w, c in vc.head(6).items())}")

# ---------------- [D] Q+1 forecast vs calendar ----------------
print("\n[D] NEXT-QUARTER FORECAST: text of quarter Q -> class of quarter Q+1 (walk-forward), vs seasonal naive")
for k in ["Apple", "Amazon"]:
    df = data[k]
    qlab = df.groupby(df.dt.dt.to_period("Q"))["class"].agg(lambda s: Counter(s).most_common(1)[0][0])
    qs = list(qlab.index); nxt = {str(q): int(qlab[qs[i + 1]]) for i, q in enumerate(qs[:-1])}
    texts, dts, _ = G[k]; qq = dts.to_period("Q").astype(str).values
    keep = np.array([q in nxt for q in qq]); Xk = X[k][keep]; qk = qq[keep]
    y1 = np.array([nxt[q] for q in qk]); uq = sorted(set(qk))
    yt, yp, ys = [], [], []
    for i, q in enumerate(uq):
        if i < 8: continue
        trm = np.isin(qk, uq[:i]); tem = qk == q
        p = clf().fit(Xk[trm], y1[trm]).predict(Xk[tem])
        # seasonal naive for Q+1: majority next-quarter label among earlier quarters with the same calendar quarter
        cal = int(q[-1]); same = [u for u in uq[:i] if int(u[-1]) == cal]
        s = Counter(int(nxt[u]) for u in same).most_common(1)[0][0] if same else 0
        yt.append(y1[tem]); yp.append(p); ys.append(np.full(tem.sum(), s))
    yt, yp, ys = map(np.concatenate, (yt, yp, ys))
    print(f"  {k:6s}: text  acc {accuracy_score(yt, yp):.3f} mcc {matthews_corrcoef(yt, yp):.3f}   |   seasonal naive  acc {accuracy_score(yt, ys):.3f} mcc {matthews_corrcoef(yt, ys):.3f}")

# ---------------- [E] tweet volume vs metric change ----------------
print("\n[E] TWEET VOLUME vs FINANCIAL CHANGE (per quarter; n is tiny, treat as a hint)")
fin = {"Apple": "appleEps.csv", "Amazon": "amazonQuarterRevenue.csv", "Tesla": "teslaCarSales.csv"}
for k in data:
    df = data[k]; vol = df.groupby(df.dt.dt.to_period("Q")).size()
    f = pd.read_csv(D + fin[k]); f["q"] = pd.to_datetime(f.from_date, dayfirst=True).dt.to_period("Q")
    f = f.set_index("q")["percent_change"].astype(float)
    common = vol.index.intersection(f.index)
    v, pc = vol[common].values.astype(float), f[common].values
    dv = np.diff(np.log(v)); dpc = pc[1:]
    r1, p1 = spearmanr(v, pc); r2, p2 = spearmanr(dv, dpc)
    print(f"  {k:6s}: n={len(common)} quarters | volume vs %change rho={r1:+.2f} (p={p1:.2f}) | volume growth vs %change rho={r2:+.2f} (p={p2:.2f})")
print("\nFINISHED")
