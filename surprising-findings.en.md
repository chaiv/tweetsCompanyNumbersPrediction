# Surprising Findings: What the Tweet Corpus Actually Contains

A summary of the exploratory probes run on August 19, 2026, after the evaluation audit
(see `evaluation-diagnosis.en.md`). The audit established what the published models do
not measure. This document records what the data turned out to contain instead - findings that were
not asked for in the thesis and that most readers would not expect.

Every number below was produced by the scripts in `tweetsCompanyNumbersPrediction/src/probes/`
on the archived labelled dataframes (groups of 10 consecutive tweets, no labels used to form the
groups, TF-IDF features and a linear classifier unless stated otherwise). Nothing here is an
interpretation of a model; all of it is a measurement.

**Creation note:** the probes and this summary were produced with Claude Fable 5; the analyses
were run directly on the repository data.

---

## Why these findings are surprising, in plain language

**Tweets are a clock.** Most people assume that a handful of short posts about a company could be
from almost any time. They cannot. Ten posts carry enough small traces - a product name, a month, a
price, an event, who posted them - to place them in the right week two times out of three. Nobody
writes the date into their tweets on purpose; the date is written in anyway.

**The clock works across companies.** One would expect that tweets about Apple and tweets about
Amazon have little in common. But a model that has only ever seen Apple tweets can still tell which
quarter an Amazon tweet comes from. The news of the day, the slang of the year and the same
automated accounts appear everywhere, so "when" is a shared property of the whole platform, not of
one company.

**The crowd is mostly machines.** The dataset is described as public opinion about five companies.
In reality, one percent of the accounts wrote more than half of all posts, and the busiest accounts
are automated news and stock-alert feeds. Whatever the models learned about "the public", they
learned largely from software.

**The published numbers were exactly as high as the calendar alone.** A rule that reads no text at
all - "this quarter of the year usually goes this way" - reaches the same accuracy and correlation
that the thesis reports for its best models. That is not a coincidence: the models had found the
calendar inside the text, because the calendar is the loudest thing in it.

**The model could not even read results that were already public.** One would expect that, at the
very least, a model could tell from tweets that Apple just reported a good quarter, since people
tweet about exactly that. It cannot, not with this kind of model. That settles the question of
whether the problem was the method or the material: the material does not carry the message in a
form such a model can read.

**Forty-two words beat 150,000.** Keeping only the names of months and seasons gives a better
forecaster for unseen quarters than the entire vocabulary does. Everything else in the text does
not help outside the period it was learned in - it actively hurts, because it anchors the model to
the wrong period.

**Being wrong in a pattern.** Out of sample, the models were not merely useless; they were worse
than guessing. The reason is almost mechanical: a model that has never seen a quarter reaches for
the most similar one, which is the quarter just before - but the financial labels flip from one
quarter to the next, so "same as last time" is reliably wrong. A coin would have done better.

**The best experiment was a game of four time windows.** The balancing step silently reduced the
flagship experiment to four windows in 2015 and 2016. The model that scored 85 percent was, in
effect, asked "is this from early 2015, autumn 2015, autumn 2016 or the holidays?" - a question the
text answers easily.

**The one thing that works is not language at all.** The only genuine signal in the whole project is
numbers that people quote in tweets about Tesla deliveries - estimates and leaked figures that
already existed before anyone tweeted them. The text is a messenger carrying a number, not a crowd
that senses something.

---

## 1. Tweets date themselves - down to the week

Ten consecutive tweets about Apple, with no label information, can be assigned to their own period:

| Resolution | Classes | Accuracy | Chance |
| --- | ---: | ---: | ---: |
| Year | 5 | 0.943 | 0.200 |
| Quarter | 20 | 0.860 | 0.050 |
| Month | 60 | 0.807 | 0.017 |
| ISO week | 262 | **0.666** | 0.004 |

Two-thirds of the time, ten tweets can be placed in the correct week out of 262. This is the
mechanism behind every high score in the thesis: the financial label is constant within a quarter,
the text identifies the quarter, and the published evaluation shared quarters between training and
test.

The clock is in the language itself, not only in automated accounts. After removing the top 1% of
accounts and every exact duplicate - which removes about 60% of all tweets - the accuracy is still
0.817 for the quarter and 0.648 for the week.

## 2. The clock transfers between companies

A quarter classifier trained only on Apple tweets dates Amazon tweets:

| Trained on | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | 0.856 | **0.664** | 0.463 |
| Amazon | 0.604 | 0.889 | 0.477 |
| Tesla | 0.494 | 0.561 | 0.818 |

Chance is 0.05. Apple to Amazon: exact quarter 66%, within one quarter 82%. After removing the top
accounts and duplicates the transfer drops to roughly half (Apple to Amazon 0.53, Amazon to Apple
0.48) but stays ten times above chance. About half of the cross-company clock is shared language;
the other half is the same automated accounts posting about several companies.

What carries the clock in raw text: explicit dates and month names, stock price levels (`114`,
`117`, `172`, `210` - the share price is a timestamp), product launches (`iphone6s`, `pixel`,
`homepod`, `iphone11`), events (`blackmonday`, `election`, `trump`), and co-mentioned tickers that
were fashionable in a given season (`nflx`, `bynd`, `roku`). The thesis pipeline strips digits and
still reaches 91.8% quarter accuracy, so the clock survives without prices.

## 3. The corpus is mostly machines talking

| Company | Authors | Top-10 accounts write | Top 1% of accounts write |
| --- | ---: | ---: | ---: |
| Apple | 89,120 | 22.9% | **67.3%** |
| Amazon | 42,512 | 14.1% | 54.4% |
| Tesla | 46,563 | 9.7% | 54.4% |

The top accounts are automated feeds: `_peripherals` and `computer_hware` (about 91,000 tweets
each), `MacHashNews`, `PortfolioBuzz`, `retail_Dbt`, `ExactOptionPick`, `TradingGuru` - and the last
three post about several companies. The dataset is described, in the thesis and on Kaggle, as
public opinion; the majority of it is algorithmic feed output.

The thesis's own interpretation results already show this. The "most important words" reported for
Apple include `cultofmac`, `DeidreZune` and `TechCrunch` - account handles. Integrated Gradients
correctly reported that the model's evidence was who posted and when; the narrative interpretation
("Apple-centric media emphasizing innovation") was added on top.

## 4. Even announced results cannot be recovered from the text

A quarter's figure is announced during the following quarter (Apple and Amazon in week 4-5, Tesla
deliveries within the first days). Tweets written in quarter Q therefore discuss the result of Q-1.
If the method could extract financial content at all, it would have to recover that. Walk-forward,
on unseen quarters only:

| Target | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Own quarter (the thesis task) | -0.042 | -0.123 | -0.011 |
| Previous quarter - already announced and discussed | **-0.134** | **-0.083** | **-0.044** |
| Next quarter (true forecast) | -0.004 | -0.111 | -0.018 |

Values are MCC. Not even the public, already-announced number is extractable from this corpus with a
bag-of-words model. The only trace: Apple's accuracy on the previous-quarter label rises from 0.21 in
week 1 of the quarter to 0.31 in week 5 - Apple's earnings week - and about 0.34 late in the quarter,
while the own-quarter accuracy stays flat. The announcement does enter the text. It is far too weak
to use.

## 5. Why every out-of-period score is negative, not zero

The full-vocabulary model's out-of-period predictions agree with the **previous** quarter's label
72% of the time (Apple) and 71% (Amazon), but with the truth only 38% and 21%. Out of period, the
model is a persistence forecaster: unseen quarters look most like the quarter just before them, so
the model returns that quarter's label.

But the labels alternate: Apple `0 0 2 3 | 0 0 2 3 | ...`, Amazon `3 0 1 1 | 3 0 1 1 | ...`.
Adjacent quarters share a label in only 32% (Apple) and 21% (Amazon) of cases. A persistence
forecaster on an alternating target is systematically wrong: MCC -0.146 and -0.176, worse than
chance. This explains every negative number in the audit, and it generalizes beyond this project:
on seasonally alternating targets, drifting-text machine learning is not neutral out of sample -
it is anti-predictive.

## 6. The only transferable content of the text is the calendar itself

Ablating the vocabulary in the walk-forward setting:

| Vocabulary (Apple) | MCC out of period |
| --- | ---: |
| Full (about 60,000 tokens) | -0.03 to -0.14 |
| Full without calendar tokens | -0.13 to -0.17 |
| Calendar tokens only | **+0.20** |

Removing calendar words makes the model worse; calendar words alone give the only positive score. A
vocabulary of just 42 month and season words reaches MCC +0.29 on Apple overall and **+0.43 on the
45% of groups that mention a month** (Amazon: +0.34 on those groups, -0.37 on the rest). Forty-two
words beat 150,000. The best honest text model is a month-name reader - and it reaches only half of
the seasonal baseline (0.76), because 55% of groups name no month at all.

## 7. Language drifts smoothly and does not recur with the seasons

Cosine similarity between quarterly TF-IDF centroids:

| Lag (quarters) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 0.894 | 0.834 | 0.789 | 0.755 | 0.729 | 0.711 | 0.693 | 0.678 |
| Amazon | 0.904 | 0.852 | 0.815 | 0.782 | 0.754 | 0.735 | 0.721 | 0.716 |
| Tesla | 0.924 | 0.882 | 0.859 | 0.841 | 0.814 | 0.794 | 0.780 | 0.768 |

The decay is monotone, about 2-3 points per quarter, Tesla slowest. Language from the same calendar
quarter one year later (lag 4) is **not** more similar than language three quarters away: there is
no seasonal recurrence in the language at the centroid level. Holiday talk is too weak to carry the
seasonal label; only explicit calendar words do. The clock is monotone: text reveals *when*, never
*which season*, from topic alone.

## 8. The documented balancing collapsed the flagship experiment onto four calendar windows

`EqualClassSampler` keeps the first n tweets per class, n being the size of the smallest class.
Applied to the temporally sorted Apple 4-class data (n = 93,686):

| Class | Contents of the balanced pool |
| --- | --- |
| 0 "decrease" | 100% tweets from 2015Q1 |
| 1 "small increase" | 100% from 2015Q3 |
| 2 "moderate increase" | 100% from 2016Q3 |
| 3 "strong increase" | 72% from 2015Q4, 28% from 2016Q4 |

The best published result (Apple@10, four classes, accuracy 0.85, MCC 0.80) therefore measured the
task "tell 2015Q1, 2015Q3, 2016Q3 and the 2015/2016 holiday seasons apart", and the model saw
essentially no tweet written after 2016Q4. The thesis states on page 114 that balancing preceded
the split; the repository contains no other balancing implementation.

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

The honest one-line description: the calendar for Apple and Amazon, delivery numbers quoted in
tweets for Tesla, plus a hand-tuned gate. The Tesla part is real (shuffling drops it from 8 to 4)
and it is the only genuine text-derived signal in the project. Its nature matters: Tesla deliveries
are a metric that the public quotes numerically before release - analyst estimates, guidance,
leaked figures - and the tweets relay those numbers. The text works as a carrier of numbers that
someone already knew, not as a source of crowd insight. EPS and revenue are not tweeted as levels,
and for them there is nothing.

## 10. What the original models learned from the word embeddings

The correlation between embedding content and label was real and strong. It ran through time:

```text
corr(embedding content, label) = corr(embedding content, period) x identity(period -> label)
```

The first factor is large - text timestamps itself to the week. The second is exact within the
sample - one quarter, one label. The trainable embedding table (63 million parameters for Apple,
against 3.8 million in the LSTM) served as a lookup from period markers to labels, with the era
structure already latent in the pretrained vectors as a head start. Out of period the second factor
breaks, the model falls back to the nearest era, and the alternating labels turn that into
anti-prediction.

## 11. What is not claimed

- Not that tweets contain no information about companies. The claim is that this corpus, these
  labels and this representation contain no extractable financial direction - not for the current
  quarter, the next one, or even the already-announced previous one.
- Not that the Tesla signal is confirmed. It rests on 4-6 quarters, p = 0.125 against the shuffle
  control, with a gate designed on the test years.
- Not that the dating accuracies are upper bounds. Stronger models would date more precisely.
- The probes used raw tweet bodies with digits; the thesis pipeline strips digits. The quarter
  result with the thesis pipeline (91.8%) is the comparable figure.

## 12. What these findings are good for

1. **A methodological result for the field.** Social-media text timestamps itself finely enough
   that any time-varying label is predictable from text without causal content. Random splits over
   time leak time. This project is an unusually clean, fully quantified case study.
2. **A reframing of the interpretation chapter.** The important-word and topic pipeline is a
   discriminative keyword analysis by period. Relabelled as "terms characteristic of each window",
   cleaned of automated accounts, and computed per quarter rather than per class, its qualitative
   observations survive; as "drivers of financial change" they do not.
3. **A direction for a confirmatory study.** The one channel that worked - numbers quoted in
   public before release - points to metrics that are publicly estimated in numeric form. That is
   a testable hypothesis with frozen features and an untouched holdout.

## 13. Proposed directions for further studies

Each direction follows from one of the findings above and is feasible with the existing data and
pipeline; the first three need no new data at all.

1. **The temporal-leakage audit as a published method.** Turn the dating probes into a general
   test: before any "text predicts X" study is believed, report (a) how precisely the text dates
   itself, (b) how well a date-only or seasonal rule predicts X, and (c) the result under
   period-grouped splits. Apply it retrospectively to published social-media prediction studies
   that used random splits. This project supplies the worked example and the numbers. (Findings 1,
   4, 5, 6)

2. **A human-only replication.** Classify accounts as automated or human (posting volume,
   duplicate rate, template regularity, posting-time regularity), then repeat every analysis on the
   human subset alone. Two questions: does any text signal appear once the feeds are gone, and how
   much of what the literature calls "public sentiment" about companies is feed output? The corpus
   share written by the top one percent of accounts should be reported in every paper that uses
   this dataset. (Finding 3)

3. **Describe the quarters properly.** Replace the neural important-word path with a per-quarter
   keyness statistic (for example weighted log-odds with a prior) on the human subset, plus topic
   prevalence over the twenty quarters. This yields the description of "what was discussed when"
   that the interpretation chapter aimed at, without a 67-million-parameter detour, and it can be
   correlated with the metrics as an explicitly exploratory n = 20 analysis. (Findings 3, 10)

4. **The numbers-as-carrier hypothesis, tested confirmatorily.** The Tesla result suggests that
   social media is useful where the public quotes a metric numerically before it is released.
   Choose metrics of that kind in advance - vehicle deliveries, unit-sales estimates, subscriber
   additions, box-office or game-sales figures, app-download counts - freeze the regexes,
   features, fusion weights and gate, and evaluate once on an untouched holdout. Compare against
   the analyst consensus directly, to learn whether the text adds anything beyond the consensus
   it relays. (Finding 9)

5. **A target with more than twenty values.** Twenty quarters per company can decide at most
   twenty cases. Either pool many companies (the same pipeline over a few hundred tickers gives
   thousands of company-quarters) or move to a target sampled more finely, such as the earnings
   surprise or the price reaction in the days after each announcement. The faint trace found in
   Apple's earnings week is exactly where an event-window design would look first. (Findings 4, 8)

6. **Predict the deviation, not the level.** Because the labels alternate seasonally, the
   informative question is not "will it go up" but "will it go up more than it usually does in
   this quarter". Define the target as the residual against the seasonal expectation; any text
   signal then has to beat zero rather than beat the calendar, and the persistence trap
   disappears by construction. (Findings 5, 6)

7. **The clock as a research object.** Measure drift rates and cross-company transfer on other
   platforms, topics and languages; separate the contributions of events, platform vocabulary and
   automated accounts; and test how far a model can date an undated post. Applications include
   provenance checking, detecting back-dated or synthetic content, and estimating the age of
   archived text. (Findings 1, 2, 7)

8. **Re-run the original hypothesis with a fair representation.** The bag-of-words and trainable-
   embedding models may simply be the wrong instruments. Repeat the period-grouped, walk-forward
   evaluation with a modern sentence encoder on the human subset and the residual target. If
   semantics beyond the period exists, this is the setting in which it could finally show; if it
   does not, the negative result becomes much stronger. (Finding 10)

## Reproduction

From `tweetsCompanyNumbersPrediction/src`, with the repository on the path:

| Script | Produces |
| --- | --- |
| `probes/exp_unexpected.py` | sections 1-3, the Q+1 forecast, tweet volume |
| `probes/exp_unexpected_control.py` | the bot-and-duplicate-removed control and the clock's carrier tokens |
| `probes/exp_deeper.py` | section 4 (lagged label, accrual by week), section 7, the announced-result vocabulary |
| `probes/exp_calendar.py` | section 6, vocabulary ablation |
| `probes/exp_calendar2.py` | section 6 (42-word model) and section 5 (persistence echo) |
| `trainNumericTextSignalQuarterModel.py` | section 9 (`output/numeric_text_signal_quarter_results.json`) |

All probes run on CPU in minutes; none uses the financial CSVs as input.
