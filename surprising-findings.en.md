# Surprising Findings: What the Tweet Corpus Actually Contains

A summary of the exploratory probes run on August 19, 2026, after the evaluation audit
(see `evaluation-diagnosis.en.md`). The audit established what the published models do
not measure. This document records what the data turned out to contain instead - findings that were
not asked for in the thesis and that most readers would not expect.

The numbers come from the scripts in `tweetsCompanyNumbersPrediction/src/probes/`, stored result
artifacts, and additional controls run on August 21, 2026, on the archived labelled dataframes. The
probes form groups of ten consecutive tweets without using labels and, unless stated otherwise, use
TF-IDF features and a linear classifier. This document explicitly separates reproduced
measurements, plausible mechanisms, and hypotheses that still require confirmation.

**Creation note:** the original probes and first summary were produced with Claude Fable 5.
ChatGPT 5.6 Sol subsequently audited the thesis, code, and measurements, reproduced the core
experiments, and added stricter cross-company controls. All local analyses ran directly on the
repository data.

---

## Why these findings are surprising, in plain language

**Tweets are a clock.** Ten posts carry enough small traces - a product name, a month, a price, an
event, who posted them - to place them in the correct one of 262 weeks two times out of three in a
mixed split that includes examples from every known week in training and test. This is not dating a
never-seen future week, but it is an unusually strong temporal fingerprint.

**The clock works across companies.** Some of the original transfer comes from identical tweets
stored in more than one company file. Yet after removing every such tweet ID and fitting the
vocabulary only on the source company, Apple-to-Amazon quarter dating still reaches 51.0% accuracy.
"When" is therefore not only a company property but also a property of the shared market and
platform discourse.

**The corpus contains a shared information-broker network.** One percent of accounts write more
than half of all posts. After identical cross-company tweets are removed, 60.3% of Apple, 86.4% of
Amazon, and 68.8% of Tesla tweets are still written by 14,143 authors active for all three
companies. Many prolific sources look like news, trading, or feed accounts, but no formal bot
classification was performed.

**The calendar dominates honest future tests.** On the same Q+1 walk-forward task, a no-text rule
reaches 83.2% accuracy/MCC 0.745 for Apple and 79.9%/0.735 for Amazon, while the linear text model
reaches only 40.2%/0.001 and 23.9%/-0.106. Because the protocol differs, these numbers are not
directly equal to the historical 87%/0.77, but they show how strong the seasonal baseline is.

**A linear full-vocabulary model does not reliably recover even announced results.** For Apple,
accuracy on the previous-quarter label visibly rises around earnings week but remains weak overall.
That is a trace of information, not proof of its absence. Stronger representations, correct temporal
alignment, and targeted numeric extraction remain open.

**For Apple, forty-two calendar words beat the full vocabulary.** Month and season words reach MCC
+0.291 walk-forward while the full vocabulary stays negative. For Amazon this holds only for groups
that actually contain a calendar word; overall MCC is nearly zero. The finding is strong but not
universal.

**The model learns a temporal persistence signal.** In the walk-forward test, 72.0% of Apple and
70.7% of Amazon predictions match the immediately preceding quarter's label. This suggests that
the full-vocabulary model mainly recognises the most recent known language regime. Because the
financial classes frequently change between adjacent quarters, this genuinely learned temporal
signal does not transfer reliably to the future target. This is not evidence of absent learning,
but of a strong signal that is misaligned with the forecasting task.

**The best experiment may have collapsed to four time windows.** If the visible
`EqualClassSampler` was used in the published run as documented, it reduced the Apple pool almost
entirely to four windows in 2015 and 2016. Without the historical run manifest this mechanism is
highly plausible, but not finally proven.

**The most promising future channel found so far is typed numeric information.** Delivery estimates
quoted in Tesla tweets provide an exploratory gain. It is not confirmed (paired exact test
p = 0.125), but it suggests a concrete architecture: metric, unit, reference period,
estimate-versus-actual status, and release time rather than word vectors alone.

---

## 1. Tweets date themselves - down to the week

Ten consecutive tweets about Apple, with no label information, can be assigned to their own period:

| Resolution | Classes | Accuracy | Chance |
| --- | ---: | ---: | ---: |
| Year | 5 | 0.943 | 0.200 |
| Quarter | 20 | 0.860 | 0.050 |
| Month | 60 | 0.807 | 0.017 |
| ISO week | 262 | **0.666** | 0.004 |

Two-thirds of the time, ten tweets can be placed in the correct one of 262 weeks already represented
in training. This demonstrates period recognition, not generalisation to a new week. It is a
plausible major mechanism behind high mixed-period results: the financial label is constant within
a quarter, the text identifies the period, and the published evaluation shared quarters between
training and test. Without the historical individual predictions, its exact contribution to any
one score cannot be determined.

The clock is not confined to the most active sources and repetitions. After removing the top 1% of
accounts and deduplicating - together removing about 60% of all tweets - accuracy remains 0.817 for
the quarter and 0.648 for the week. Because this control changes activity, author identity, and
duplicates at once, it does not isolate a pure bot contribution.

## 2. The clock transfers between companies

A quarter classifier trained only on Apple tweets dates Amazon tweets:

| Trained on | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | 0.856 | **0.664** | 0.463 |
| Amazon | 0.604 | 0.889 | 0.477 |
| Tesla | 0.494 | 0.561 | 0.818 |

Chance under uniform classes is 0.05. Apple to Amazon reaches 66% exact-quarter accuracy and 82%
within one quarter. This first test, however, fitted a shared vocabulary across all companies, and
8.6-17.0% of tweet IDs overlap for each company pair. It does not yet establish clean transfer.

A stricter control removed every tweet ID appearing in more than one company file and fitted the
vocabulary only on the source company:

| Trained on | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | - | **0.510** | 0.311 |
| Amazon | **0.489** | - | 0.344 |
| Tesla | 0.360 | **0.423** | - |

A strong shared platform and market state therefore remains. After cross-company duplicate removal,
60.3% of Apple, 86.4% of Amazon, and 68.8% of Tesla tweets come from 14,143 authors active for all
three companies. Removing every author who appears in multiple company sets leaves several
directions above their target-company majority baseline: Apple to Amazon 0.204 versus 0.083, Amazon
to Apple 0.166 versus 0.112, and Tesla to Amazon 0.206 versus 0.083. Other directions lie closer to
baseline. This supports two layers: a dominant shared information-broker network and a weaker,
more general language of the era.

What carries the clock in raw text: explicit dates and month names, stock price levels (`114`,
`117`, `172`, `210` - the share price is a timestamp), product launches (`iphone6s`, `pixel`,
`homepod`, `iphone11`), events (`blackmonday`, `election`, `trump`), and co-mentioned tickers that
were fashionable in a given season (`nflx`, `bynd`, `roku`). A historical 91.8% quarter-recognition
rate is reported but cannot be reproduced from the retained artifacts. The current diagnostic,
which fits its TF-IDF vocabulary only on training groups, reaches 0.730 accuracy with seed 1337. The
clock remains strong; 91.8% must not be presented as currently confirmed.

## 3. A small number of professional sources dominate the corpus

| Company | Authors | Top-10 accounts write | Top 1% of accounts write |
| --- | ---: | ---: | ---: |
| Apple | 89,120 | 22.9% | **67.3%** |
| Amazon | 42,512 | 14.1% | 54.4% |
| Tesla | 46,563 | 9.7% | 54.4% |

Top accounts include `_peripherals` and `computer_hware` (about 91,000 tweets each), `MacHashNews`,
`PortfolioBuzz`, `retail_Dbt`, `ExactOptionPick`, and `TradingGuru`; several post about multiple
companies. The thesis itself notes that its ten most frequent sources are financial-news accounts
rather than typical individuals. The new result is therefore not their mere existence but their
extreme concentration and cross-company network. Individual accounts were not classified as bots,
partly automated feeds, or manually operated professional sources.

The thesis's own interpretation results already show this. The "most important words" reported for
Apple include `cultofmac`, `DeidreZune` and `TechCrunch` - account handles. Integrated Gradients
therefore shows that a prediction can respond to source markers. The attribution explains what the
model reacted to; it proves neither that the account explains the financial change nor that the
subsequent economic narrative is causal.

## 4. A linear text model does not reliably recover even announced results

A metric is normally announced after its economic reporting quarter has ended. Tweets written in
quarter Q can therefore discuss Q-1. As a negative control, a linear TF-IDF model was tested on its
ability to recover this already-public label walk-forward:

| Target | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Own quarter (the thesis task) | -0.042 | -0.123 | -0.011 |
| Previous quarter - already announced and discussed | **-0.134** | **-0.083** | **-0.044** |
| Next quarter (true forecast) | -0.004 | -0.111 | -0.018 |

Values are MCC. The tested bag-of-words model does not reliably recover the public previous-quarter
label. A trace remains: Apple's accuracy on the previous-quarter label rises from 0.21 in
week 1 of the quarter to 0.31 in week 5 - Apple's earnings week - and about 0.34 late in the quarter,
while own-quarter accuracy stays flat. The announcement therefore enters the text but is only weakly
extracted by this representation. Because the TF-IDF vectorizer was fitted on the full period before
the folds and only one model family was tested, this is not a general impossibility result.

## 5. Why a learned time signal can produce negative MCC values

In the walk-forward test, the full-vocabulary model's predictions agree with the **previous**
quarter's label in 72.0% of Apple and 70.7% of Amazon cases. They agree with the correct class in
only 37.8% and 20.5% of cases. What is measured is therefore a strong previous-quarter resemblance
in the predictions. The further explanation that the model primarily recognises the most recent
known language regime is a plausible mechanism, not direct evidence of nearest-neighbour behaviour.

A separately constructed pure persistence rule assigns every group the previous quarter's class.
It reaches accuracy 0.298/MCC -0.146 for Apple and 0.165/-0.176 for Amazon. This follows from the
target structure: adjacent quarters share a class in only 32% of Apple and 21% of Amazon cases. A
useful temporal signal therefore becomes a poor financial signal when the target classes change
seasonally.

"Worse than guessing" would still be too broad. With four classes, a uniformly random predictor
has expected accuracy of 25% and MCC near zero. Apple's full-vocabulary accuracy of 37.8% exceeds
that simple accuracy reference, whereas Amazon's 20.5% falls below it. The negative MCC values above
also belong to the pure persistence rule, not to an exact identification of the full-vocabulary
model with that rule. An MCC below zero here means an anti-predictive mapping relative to the actual
class structure; it does not mean that every arbitrary random prediction would beat the model on
every metric.

The finding therefore does not show that the model learns nothing. It shows that the model learns a
strong and interpretable signal that is misaligned with future forecasting. Because this probe
fitted its TF-IDF vectorizer on the full period in advance, the exact values should additionally be
confirmed with per-fold fitting. The broader hypothesis remains that an era-similarity classifier
trained on drifting text can become anti-predictive out of sample when the target changes
seasonally.

## 6. Calendar words are the most stable transferable text content tested

Ablating the vocabulary in the walk-forward setting:

| Vocabulary (Apple) | MCC out of period |
| --- | ---: |
| Full (about 60,000 tokens) | -0.03 to -0.14 |
| Full without calendar tokens | -0.13 to -0.17 |
| Calendar tokens only | **+0.20** |

Removing calendar words makes the Apple model worse; calendar words alone give the only positive
score in that ablation. A fixed vocabulary of 42 month and season words reaches MCC +0.29 for Apple
overall and **+0.43 on the 45% of groups containing such a word**. For Amazon, overall MCC is only
+0.005; it rises to +0.34 on groups with a calendar word and falls to -0.37 on the rest. For Apple,
42 fixed words therefore beat the roughly 60,000-token full vocabulary. Even there the model
captures only part of the pure seasonal baseline (MCC 0.76), because 55% of groups contain no
calendar word.

## 7. At centroid level, language drifts smoothly without a seasonal peak

Cosine similarity between quarterly TF-IDF centroids:

| Lag (quarters) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 0.894 | 0.834 | 0.789 | 0.755 | 0.729 | 0.711 | 0.693 | 0.678 |
| Amazon | 0.904 | 0.852 | 0.815 | 0.782 | 0.754 | 0.735 | 0.721 | 0.716 |
| Tesla | 0.924 | 0.882 | 0.859 | 0.841 | 0.814 | 0.794 | 0.780 | 0.768 |

The decay is monotone, about 2-3 points per quarter, Tesla slowest. Language from the same calendar
quarter one year later (lag 4) is **not** more similar than language three quarters away: there is no
additional seasonal peak at the centroid level. In this coarse representation the clock is mostly
monotone: text strongly reveals *when*, while recurrence of the entire topic language after four
quarters is not visible. This does not exclude weaker seasonal subtopics.

## 8. The documented balancing can reduce the flagship experiment to four windows

`EqualClassSampler` keeps the first n tweets per class, n being the size of the smallest class.
Applied to the temporally sorted Apple 4-class data (n = 93,686):

| Class | Contents of the balanced pool |
| --- | --- |
| 0 "decrease" | 100% tweets from 2015Q1 |
| 1 "small increase" | 100% from 2015Q3 |
| 2 "moderate increase" | 100% from 2016Q3 |
| 3 "strong increase" | 72% from 2015Q4, 28% from 2016Q4 |

If the published Apple run used this sampler as described on page 114, its best result (Apple@10,
four classes, accuracy 0.85, MCC 0.80) essentially measured "tell 2015Q1, 2015Q3, 2016Q3, and the
2015/2016 holiday seasons apart"; the pool then contained almost no tweet after 2016Q4. The
repository contains no other visible balancing implementation. Without the historical run manifest,
exact commit, and intermediate artifact, the assignment is highly plausible but conditional.

## 9. The new hybrid model, decomposed: calendar plus quoted numbers

The hybrid on this branch (`trainNumericTextSignalQuarterModel.py`) reports 80.56% accuracy and
MCC 0.7387 on 36 later company-quarters. The run reproduces to four decimals. Per company, correct
out of 12:

| Branch | Amazon | Apple | Tesla | Pooled |
| --- | ---: | ---: | ---: | ---: |
| Seasonal prior only | 10 | 10 | 2 | 22/36 |
| + numeric text (50/50) | 10 | 11 | 4 | 25/36 |
| + Tesla forward-level branch | 10 | 11 | 6 | 27/36 |
| + Tesla conflict gate (post hoc) | 10 | 11 | **8** | **29/36** |
| + gate, with shuffled text | 10 | 11 | 4 | 25/36 |

Of the seven quarters gained over the calendar, six are Tesla, two of those come from a gate tuned
after inspecting the test years, and Amazon gains none. On direction (increase versus decrease) the
seasonal prior alone scores 0.9167 / 0.8003 - identical to the full hybrid. The general hybrid with
shuffled text (MCC 0.5473) slightly beats the same hybrid with real text (0.5466).

The honest short description is the calendar for Apple and Amazon, delivery numbers quoted in Tesla
tweets, plus a hand-tuned gate. Shuffling the text features reduces Tesla from 8 to 4 correct
quarters; relative to the paired shuffle control there are four cases solved only by real text and
none in the opposite direction. The exact two-sided test is nevertheless p = 0.125, and the gate was
designed after inspecting the test years. The Tesla result is therefore promising but unconfirmed.

Its nature is scientifically important: Tesla deliveries are numerically estimated before release.
Tweets can relay analyst estimates, guidance, or other values circulating in advance. Text is then
not merely sentiment but a transport channel for structured numeric information. Whether this
channel adds value beyond analyst consensus on a frozen holdout is the decisive open question.

### Target timing: Amazon is aligned differently

The Amazon finance CSV is shifted one quarter forward relative to the economic reporting quarter:
USD 29.33 billion from Amazon Q4 2014 is stored as 2015Q1, while USD 22.72 billion from Q1 2015 is
stored as 2015Q2. Consistently, the thesis explains that the values labelled Q1 are highest because
of holiday sales in the previous quarter. Tweets in the nominal Amazon target quarter can therefore
already contain the release of that target value. The task is partly report reaction or
reconstruction, not exclusively forecasting.

Future evaluation must store three times for every target: economic reporting quarter, release
timestamp, and permitted forecast cutoff. The target can remain the quarterly number; only the
information available before prediction must be bounded correctly.

## 10. What the original models learned from the word embeddings

The correlation between embedding content and label was real and strong in the mixed split. As a
mechanism sketch, not a mathematical correlation identity, it can be written as:

```text
embedding content -> period -> fixed within-dataset quarter label
```

The first link is strong - text finely timestamps itself within known periods. The second is
deterministic within the sample - one quarter, one label. The trainable embedding table (63 million
parameters for Apple, against 3.8 million in the LSTM) could serve as a lookup from period markers to
labels, potentially helped by era structure already latent in the pretrained vectors. Out of period,
that mapping can break; the observed fallback to the preceding era and alternating labels then
explain the anti-prediction of the linear controls. In addition, the historical Word2Vec
representations were trained on the full corpus before cross-validation and were therefore
transductive.

### What the thesis's third step can genuinely contribute

The topic/important-word stage remains scientifically valuable, but in a different role from its
original interpretation. Many manual topics - Brexit, elections, COVID-19, or the Hong Kong
protests - are one-off time anchors. A topic-to-class association can therefore again represent
`topic -> period -> quarter label` rather than an economic pathway. Integrated Gradients also
explains the model prediction, not the cause of the real-world metric.

The document and implementation do not fully agree on selection either. The thesis describes ten
particularly similar tweet groups; the visible code retrieves ten similar learned topics, takes all
documents assigned to them, and then forms label-dependent groups. Some narratives are constructed
retrospectively; the clearest anachronism is explaining tweets from a corpus ending in 2020 through
the `#AppleToo` movement that began in 2021.

Reframed positively, this stage can produce a dynamic event and source atlas: past-only topics per
quarter, topic prevalence over time, separate professional and individual sources, and important
words as descriptions of model reaction. Only a subsequent change in topic prevalence tested on
untouched quarters should become a forecast candidate.

## 11. What is not claimed

- Not that tweets contain no information about companies. The evidence is only that the tested
  linear full-vocabulary models did not extract stable financial direction walk-forward. The Apple
  earnings-week trace and Tesla numeric signal argue against a general absence of information.
- Not that the Tesla signal is confirmed. It rests on 4-6 quarters, p = 0.125 against the shuffle
  control, with a gate designed on the test years.
- Not that the dating accuracies are upper bounds. Stronger models would date more precisely.
- Not that the most active accounts have been proven to be bots. Concentration, source overlap, the
  news/feed character of many top accounts, and duplicates are established.
- Several TF-IDF probes fitted vocabulary and IDF on the full period in advance. The fixed
  42-calendar-word model is unaffected; exact full-vocabulary walk-forward numbers are transductive
  and should be repeated with per-fold fitting.
- The historical 91.8% quarter-recognition rate is reported but not reproduced. The current
  training-only-fit diagnostic reaches 73.0% with seed 1337.

## 12. What these findings are good for

1. **A methodological result for the field.** Social-media text timestamps itself finely enough
   that a time-varying label can appear predictable from text under randomly time-mixed splits even
   when the text contains no causal content about the target. Random splits over
   time leak time. This project is an unusually clean, fully quantified case study.
2. **A reframing of the interpretation chapter.** The important-word and topic pipeline is a
   discriminative keyword analysis by period. Relabelled as "terms characteristic of each window",
   stratified by source and computed per quarter rather than per class, its qualitative
   observations survive; as "drivers of financial change" they do not.
3. **A direction for a confirmatory study.** The channel with an exploratory positive result -
   numbers quoted in public before release - points to metrics that are publicly estimated in
   numeric form. This is a testable but still unconfirmed hypothesis with frozen features and an
   untouched holdout.
4. **A new research object: the information-broker network.** The large author overlap shows that
   the corpus captures a shared market information infrastructure. Source reliability, lead-lag
   relations, news propagation, and cross-company regimes can be studied independently of whether
   a quarterly forecast succeeds.
5. **A positive architecture principle.** The strong time encoder need not be discarded. It can be
   modelled as an explicit nuisance branch, while a second representation made as invariant as
   possible to time and source predicts the quarterly residual. Only the second branch's incremental
   value would count as credible financial text signal.

## 13. Proposed directions for further studies

Each direction follows from the findings above. Time, source, topic, and representation controls are
feasible with the existing data. Exact release timestamps, analyst consensus, or additional
companies require added metadata or financial series; the target remains a quarterly metric in all
forecast variants.

1. **The temporal-leakage audit as a published method.** Turn the dating probes into a general
   test: before any "text predicts X" study is believed, report (a) how precisely the text dates
   itself, (b) how well a date-only or seasonal rule predicts X, and (c) the result under
   period-grouped splits. Apply it retrospectively to published social-media prediction studies
   that used random splits. This project supplies the worked example and the numbers. (Findings 1,
   4, 5, 6)

2. **A source-stratified replication.** Classify accounts as automated, professional, or individual
   (posting volume, duplicate rate, template regularity, posting-time regularity), then repeat every analysis on the
   human subset alone. Two questions: does any text signal appear once the feeds are gone, and how
   much of what the literature calls "public sentiment" about companies is feed output? The corpus
   share written by the top one percent of accounts should be reported in every paper that uses
   this dataset. (Finding 3)

3. **Model the information-broker network.** Treat the 14,143 authors active for Apple, Amazon, and
   Tesla as a graph: who publishes a metric or estimate first, who relays it, how stable is each
   source, and how quickly does information move between companies? Identical tweet IDs must be
   removed before every cross-company test. An author holdout tests transfer to unseen sources.
   (Findings 2, 3)

4. **Describe the quarters properly.** Replace the neural important-word path with a per-quarter
   keyness statistic (for example weighted log-odds with a prior) on the human subset, plus topic
   prevalence over the twenty quarters. This yields the description of "what was discussed when"
   that the interpretation chapter aimed at, without a 67-million-parameter detour, and it can be
   correlated with the metrics as an explicitly exploratory n = 20 analysis. (Findings 3, 10)

5. **The numbers-as-carrier hypothesis, tested confirmatorily.** The Tesla result suggests that
   social media is useful where the public quotes a metric numerically before it is released.
   Choose metrics of that kind in advance - vehicle deliveries, unit-sales estimates, subscriber
   additions, box-office or game-sales figures, app-download counts - freeze the regexes,
   features, fusion weights and gate, and evaluate once on an untouched holdout. Compare against
   the analyst consensus directly, to learn whether the text adds anything beyond the consensus
   it relays. (Finding 9)

6. **Measure an information curve relative to release.** Group tweets not only by calendar quarter
   but by distance from the release: 30 and 7 days before, release day, and after. This separates
   expectation, announcement, and reaction. Forecast evaluation may use only pre-release text. The
   Amazon timing shift makes this control mandatory. (Findings 4, 9)

7. **A target with more than twenty values.** Twenty quarters per company can decide at most
   twenty cases. Either pool many companies (the same pipeline over a few hundred tickers gives
   thousands of company-quarters) or move to a target sampled more finely, such as the earnings
   surprise or the price reaction in the days after each announcement. The faint trace found in
   Apple's earnings week is exactly where an event-window design would look first. (Findings 4, 8)

8. **Predict the deviation, not the level.** Because the labels alternate seasonally, the
   informative question is not "will it go up" but "will it go up more than it usually does in
   this quarter". Define the target as the residual against the seasonal expectation; any text
   signal then has to beat zero rather than beat the calendar, and the persistence trap
   disappears by construction. (Findings 5, 6)

9. **The clock as a research object.** Measure drift rates and cross-company transfer on other
   platforms, topics and languages; separate the contributions of events, platform vocabulary and
   automated accounts; and test how far a model can date an undated post. Applications include
   provenance checking, detecting back-dated or synthetic content, and estimating the age of
   archived text. (Findings 1, 2, 7)

10. **Separate time/source representation from the financial residual.** Train a shared encoder on
    all five companies to represent time, events, and source. An adversarial or orthogonalised
    second branch should remove time and author information and predict only the incremental value
    over the seasonal quarterly baseline. Google and Microsoft tweets can serve as unlabelled
    control domains; the targets for the three forecast companies remain their quarterly numbers.
    (Findings 1–3, 7)

11. **Re-run the original hypothesis with a fair representation.** The bag-of-words and trainable-
   embedding models may simply be the wrong instruments. Repeat the period-grouped, walk-forward
   evaluation with a modern sentence encoder on the human subset and the residual target. If
   semantics beyond the period exists, this is the setting in which it could finally show; if it
   does not, the negative result becomes much stronger. (Finding 10)

12. **Study the aggregation scale itself.** N = 10 frequently outperforming 5 or 20 across models
    and companies may reveal a real context-to-noise optimum. Compare count-based groups with fixed
    time windows and length-matched controls before deciding whether N = 10 is linguistic,
    temporal, or merely technical.

## Reproduction

From `tweetsCompanyNumbersPrediction/src`, with the repository on the path:

| Script | Produces |
| --- | --- |
| `probes/exp_unexpected.py` | sections 1-3, the Q+1 forecast, tweet volume |
| `probes/exp_unexpected_control.py` | the top-one-percent-and-duplicate-removed control and the clock's carrier tokens |
| `probes/exp_cross_company_network.py` | cross-company ID, author-network, and source-exclusive transfer controls from section 2 |
| `probes/exp_deeper.py` | section 4 (lagged label, accrual by week), section 7, the announced-result vocabulary |
| `probes/exp_calendar.py` | section 6, vocabulary ablation |
| `probes/exp_calendar2.py` | section 6 (42-word model) and section 5 (persistence echo) |
| `trainNumericTextSignalQuarterModel.py` | section 9 (`output/numeric_text_signal_quarter_results.json`) |

The probes run on CPU; the large TF-IDF controls can require several minutes. On Windows, enable
UTF-8 output for `exp_deeper.py`, because some tokens cannot be represented in CP1252. Only the
tweet-volume analysis in `exp_unexpected.py` loads the financial CSVs; the dating, source, and
vocabulary findings do not. Forecast targets in the other probes come from the already-labelled
dataframes.
