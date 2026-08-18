# What the Old Models Learned—and What the Current Results Actually Rest On

A comprehensive, plain-language evaluation of the old `main` implementation, the earlier automated analysis, the current quarterly models, and the topic and important-word analysis.

Status: August 18, 2026

**Creation note:** The diagnosis, code review, and subsequent model training were performed automatically using ChatGPT 5.6 Sol and Claude Fable / Opus 5.

## Executive Summary

This project investigates whether public texts about companies can be used to infer changes in a quarterly business metric. Examples include Amazon revenue, Apple EPS, and Tesla deliveries. It also aims to explain which words and topics are associated with the model's decisions.

The main findings are:

1. **The old implementation is scientifically interesting.** It represents an unusually broad information system: tweets are linked to reporting periods, aggregated into groups, classified with an LSTM, and then interpreted down to words and topics. The strong temporal and event-related fingerprint in the language is particularly interesting.
2. **The old accuracy of approximately 0.87 is not clean evidence of genuine future forecasting.** Many tweet groups from the same quarter were treated as if they were independent test cases. In addition, the main evaluation could train on later text blocks when testing an earlier block. The model could therefore recognize periods and sources without having to predict an unknown later quarterly value.
3. **The earlier automated analysis correctly identified the central evaluation weakness.** Some statements were nevertheless too absolute or factually imprecise. There are not 20 different class labels, for example, but 20 quarterly outcomes with only two or four possible classes. The entire interpretation path was not automatically correct either.
4. **The current best result is 80.56% accuracy and MCC 0.7387 on 36 later company-quarters.** This result belongs to a hybrid of a seasonal prior, numerical text signals, and a Tesla-specific branch. It is not a pure-text model.
5. **The isolated numerical text branch reaches 50.00% accuracy and MCC 0.3224.** The transparent variant without the subsequently designed Tesla conflict gate reaches 75.00% accuracy and MCC 0.6633.
6. **The 80.56% result is exploratory.** The Tesla conflict gate was designed after inspecting errors from the same 2017–2019 test years. It must be confirmed on new, previously untouched quarters.
7. **Topics and important words are now connected in a more temporally valid way.** Exact model attributions are separated from descriptive topic explanations. Topic models and word lexicons are fitted only on earlier quarters and then applied to future quarters.
8. **The branch is not yet completely free of tweet content.** Large raw datasets are not committed, but some demo and test files still contain complete or tweet-like texts.

The scientifically accurate overall statement is therefore:

> The old work provides a broad, explainable research platform and strong temporal language signatures. The current system provides exploratory evidence that local text signals can be useful in a leakage-aware quarterly hybrid. Neither the old 87% accuracy nor the current 80.56% yet demonstrates a confirmed pure-text model for unknown future quarters.

---

## Table of Contents

1. [The Research Question in Plain Language](#1-the-research-question-in-plain-language)
2. [Key Terms](#2-key-terms)
3. [Scope and Standard of Evidence](#3-scope-and-standard-of-evidence)
4. [The Old `main` Pipeline Step by Step](#4-the-old-main-pipeline-step-by-step)
5. [Why the Old Evaluation Could Produce High Scores](#5-why-the-old-evaluation-could-produce-high-scores)
6. [What Is Scientifically Interesting About the Old Implementation](#6-what-is-scientifically-interesting-about-the-old-implementation)
7. [Weak and Faulty Parts of `main`](#7-weak-and-faulty-parts-of-main)
8. [Review of the Earlier Automated Analysis](#8-review-of-the-earlier-automated-analysis)
9. [The Current Quarterly Model](#9-the-current-quarterly-model)
10. [Training and Future Testing](#10-training-and-future-testing)
11. [Results and Statistical Interpretation](#11-results-and-statistical-interpretation)
12. [Topics and Important Words](#12-topics-and-important-words)
13. [Concrete Explanation Examples](#13-concrete-explanation-examples)
14. [Claim Ladder](#14-claim-ladder)
15. [Privacy and Tweet-Content Audit](#15-privacy-and-tweet-content-audit)
16. [Recommended Research Roadmap](#16-recommended-research-roadmap)
17. [Reproduction and Code Evidence](#17-reproduction-and-code-evidence)
18. [Final Assessment](#18-final-assessment)

---

## 1. The Research Question in Plain Language

### 1.1 What Is Being Predicted?

The target remains exclusively a **quarterly business metric**, or its change. A stock price is not used as a substitute target.

The current multi-company experiment uses:

| Company | Metric |
| --- | --- |
| Amazon | Revenue, or net sales |
| Apple | Earnings per share, or EPS |
| Tesla | Vehicle deliveries, or car sales |

The percentage change is divided into four classes:

| Class | Meaning |
| ---: | --- |
| 0 | Decrease |
| 1 | Small increase from 0% to 15% |
| 2 | Moderate increase above 15% and up to 30% |
| 3 | Large increase above 30% |

The direction—decrease versus increase—can also be evaluated. It remains a simpler supplementary diagnostic and does not replace the four-class target.

### 1.2 What Information May the Model Use?

The current text system uses only locally available texts and aggregations computed from them. The finance CSV and the current quarterly value being predicted are not used as input features.

Past target classes may be used for training and for a seasonal prior. This is important: a model without current financial values can still know historical target labels. The correct description of the best system is therefore **no-finance hybrid**, not **pure text**.

### 1.3 Is This a Forecast of the Next Quarter?

Not yet in the strict `Q -> Q+1` sense.

Texts from a quarter are used to estimate the metric for that same reporting quarter. Because the metric is typically reported only after the quarter ends, this is a **current-quarter nowcast**.

At the same time, the temporal test is substantially cleaner: the model is trained on earlier years and evaluated on later years. Therefore:

- **Temporally future test:** yes.
- **Target is the next quarter, Q+1:** no.
- **Target is the text's own quarter:** yes.

A genuine Q+1 experiment would be possible, but it would require shifting the targets by one quarter and would answer a different research question.

---

## 2. Key Terms

### Company-Quarter

A combination of a company and a quarter, such as `Amazon 2018Q2`. All texts in this combination refer to the same target realization. A company-quarter is therefore the primary independent unit of evaluation.

### Training, Validation, and Test

- **Training:** The data from which the model learns its parameters.
- **Validation:** The data used to select the model variant and hyperparameters.
- **Test:** The data that may be used for final evaluation only after selection is complete.

### Leakage

Leakage means that information from a test case enters training or model selection, directly or indirectly. This can happen even without identical texts. If other texts from the same quarter are in training, the model already knows many characteristics of the exact period it is supposedly predicting.

### Pseudoreplication

A quarter has only one financial target. If one thousand tweet groups from that quarter are counted as one thousand independent test cases, the same target realization is repeated one thousand times. The sample then appears larger and more certain than it really is.

### Accuracy

The proportion of correctly predicted classes. Here, 80.56% means 29 correct decisions out of 36 company-quarters.

### MCC

The Matthews correlation coefficient accounts for all parts of the confusion matrix and is more informative than accuracy alone when classes are imbalanced. A value of 1 is perfect, 0 roughly indicates no association, and negative values suggest systematically incorrect decisions.

### Log Loss

Log loss evaluates not only the selected class but also the model's confidence. A confidently wrong prediction is penalized more heavily.

### Baseline

A simple comparison rule. A complex text model should, for example, outperform a seasonal rule that considers only earlier instances of the same calendar quarter.

### Shuffle Control

Text signals are deliberately exchanged between quarters while targets and the other model components remain unchanged. If performance does not fall, the text was probably not decisive.

### Attribution

An attribution describes how strongly a particular input feature contributes to a model decision. It explains the model, but it does not establish an economic cause.

### Topic

A topic is a group of terms that frequently occur together. A topic summarizes textual context. It is not automatically a causal reason for a quarterly change.

---

## 3. Scope and Standard of Evidence

### 3.1 Examined Revisions

| Examined item | Reference |
| --- | --- |
| Old implementation | `main` at commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6` |
| Current branch | `baselines`, HEAD `0e708b1bc5a4a58c75c27f5f6ccb40e8a2f3e9bf`, plus the current working tree |
| Primary result | `output/numeric_text_signal_quarter_results.json` |
| Topic/word result | `output/numeric_text_topics_important_words.json` |
| Test period | Rolling tests for 2017, 2018, and 2019 |

The `main` branch was read directly from Git objects. No checkout was needed, so the already modified working tree remained untouched.

### 3.2 Three Levels of Evidence

| Level | Meaning | Example |
| --- | --- | --- |
| Code fact | Directly visible in the referenced code | `pd.to_datetime(post_date)` is called without `unit='s'`. |
| Reproduced result | Targets, probabilities, and metrics are available locally and were recomputed | 29 of 36 company-quarters correct, accuracy 0.8056, MCC 0.7387. |
| Reported historical finding | Reported only in the earlier analysis or dissertation | The exact historical quarter-recognition rate cannot be fully reproduced without its original result artifact. |

### 3.3 Why There Are Only 36 Primary Test Cases

There are three companies, four quarters per year, and three test years:

```text
3 companies x 4 quarters x 3 test years = 36 company-quarters
```

Whether 1,000 or 1,000,000 texts are processed internally does not change this number. More texts can improve a quarter representation, but they do not create new independent financial events.

### 3.4 Limitation of This Audit

The audit evaluates the visible repository state. It does not reconstruct the exact hardware, data revision, and checkpoint selection of every published historical run. It is therefore impossible to determine with certainty which old checkpoint produced every published number.

---

## 4. The Old `main` Pipeline Step by Step

### 4.1 Data and Experiment Registry

`PredictionModelPath.py` defines experiments for several companies, metrics, group sizes, and target variants. The code includes, among others:

- Amazon revenue,
- Apple EPS,
- Tesla car sales,
- Google search-engine market share,
- binary classes,
- four classes,
- group sizes of 5, 10, and 20.

This is scientifically interesting because the same overall idea can be applied to different companies and types of metrics.

### 4.2 Linking Tweets and Financial Values

`TweetNumbersConnector` searches for the financial row whose time interval contains a tweet's timestamp. It enforces two useful integrity rules:

- If no matching financial row exists, the process stops.
- If multiple financial rows match, the process also stops.

The tweet therefore receives the value from its own reporting period. Parts of the old README, by contrast, described the next reported value. The code and documentation thus did not describe the same forecast horizon.

### 4.3 Discretizing the Target

The four-level path distinguishes a decrease and small, moderate, and large increases. The old classifier has several technical boundary issues:

- The intervals overlap at 15 and 30.
- Because the first matching rule wins, exact values of 15 and 30 enter the lower class.
- There is a small gap between -0.01 and 0.
- The binary and four-class paths do not always use the same scale: one uses a ratio and the other a percentage value.

The target schema should therefore be explicitly versioned and protected by boundary-value tests.

### 4.4 Forming Tweet Groups

In simplified terms, `DataframeSplitter.getSplitIds` works as follows:

1. All rows belonging to a class are selected.
2. These rows are cut into consecutive blocks of size 5, 10, or 20.
3. Each block becomes a training sample.
4. All texts in the block receive the same class label.

The grouping logic has no explicit quarter boundary. At a period boundary, texts from different quarters can therefore enter the same group if they share the same class.

More importantly, it repeats the target: a single quarter can produce many groups even though all those groups share one financial realization.

### 4.5 Representing a Group

`TweetGroup` tokenizes each post and joins the token sequences with a `<SEP>` token. This is a sensible idea because tweet boundaries do not disappear completely.

The group is then passed to the LSTM as one long sequence. A true `token -> tweet -> quarter` hierarchy does not yet exist in the old model.

### 4.6 Top2Vec and Word Vectors

Top2Vec is trained on the local text corpus. Its word vectors are used as a 300-dimensional initialization for the LSTM. The same semantic space therefore serves both prediction and topic interpretation.

This is conceptually elegant but problematic for a strict future test: if Top2Vec is trained on the entire corpus before the split, later test texts already affect the vocabulary and vectors. This is not direct label leakage, but it is transductive use of future texts.

### 4.7 The Old LSTM

The main model contains:

- a trainable embedding,
- a two-layer LSTM with hidden size 512,
- two hidden linear layers,
- a final class output.

The last hidden state is used to represent the entire padded sequence. In principle, this can work with correctly packed sequences. The old code, however, lacks true sequence lengths, packing, and masking. The final state can therefore pass through many PAD steps.

### 4.8 Training

The Lightning trainer uses:

- CUDA by default,
- mixed precision,
- TensorBoard logging,
- checkpointing by validation loss,
- early stopping,
- optional class weights.

This infrastructure is a strength. Some details remain problematic: the early-stopping patience is as large as the maximum number of epochs, and manually reconstructed checkpoint paths can load an older unversioned file.

### 4.9 Old Primary Evaluation

The primary evaluation divides the groups of each class into ten chronological blocks. For fold `k`:

- Block `k` is the test set.
- All other blocks become training and validation data.

For an early test block, later blocks can therefore be in training. Other groups from the same quarter can also appear on both sides.

The evaluation is consequently not a strict forecast of later unknown quarters.

### 4.10 Other Split Designs

The codebase also contains:

- a global 80/20 temporal split,
- a per-class expanding-window variant,
- a stratified-temporal variant,
- a subsequent variant for interpretation.

Implementing several protocols demonstrates the right scientific concern: results should be tested against different temporal assumptions. The old implementations do not, however, automatically eliminate quarterly pseudoreplication, target stratification, or the faulty time conversion.

### 4.11 Old Interpretation Pipeline

The old idea was:

1. Integrated Gradients computes token attributions.
2. Tokens are mapped back to their tweets.
3. The original word and part-of-speech tag are added.
4. Top2Vec or BERTopic assigns documents to topics.
5. Important words and topics are analyzed together.
6. Manually or LLM-generated topics can be compared with model topics.

This research logic is strong. Several implementation details were faulty, however; they are described in Section 7.

---

## 5. Why the Old Evaluation Could Produce High Scores

### 5.1 A Synthetic Example

Suppose Tesla 2018Q4 has class 3 and there are 10,000 texts. With a group size of 10, this produces approximately 1,000 groups with the same label.

If a split tests 100 groups and trains on 900 groups from the same quarter, the tweet IDs are different. The model nevertheless sees many other texts from that exact period.

It can therefore learn:

| Text characteristic | What it can reveal |
| --- | --- |
| A product name used at the time | Quarter or product cycle |
| A campaign active at the time | Period |
| A particular news source | Source and publication phase |
| Market terms typical of the period | Market regime |
| A recurring template | Source or period |

Because the prepared dataset assigns a fixed label to the quarter, recognizing the period can already be sufficient for apparently strong financial classification.

### 5.2 What the High Accuracy Then Actually Measures

The old accuracy can measure a mixture of:

- genuine financial information in the text,
- recognition of the quarter,
- recognition of the year,
- seasonal patterns,
- source and author patterns,
- product and event names,
- recurring text templates.

Without appropriate baselines and a global future split, it is impossible to determine what proportion is genuine forward-relevant textual information.

### 5.3 Why “20 Distinct Labels” Is Wrong

From 2015Q1 through 2019Q4, there are 20 quarterly realizations per company. But there are only four possible class values, or two for the binary target.

The correct statement is:

> Many tweet groups share only 20 independent quarterly outcomes.

The following statement is wrong:

> There are 20 distinct class labels.

### 5.4 The Seasonal Baseline

Many company metrics have recurring quarterly patterns. Q4 may regularly look different from Q1, for example.

For a new Q2, a seasonal baseline asks only: which classes did earlier Q2 observations from the same company have?

If this simple rule is already strong, a text model must show what additional value the text contributes. A comparison only against the global majority label is not sufficient.

---

## 6. What Is Scientifically Interesting About the Old Implementation

The methodological weaknesses do not make the old work worthless. They change which conclusions are justified.

### 6.1 A Complete Information System

The codebase implements more than an LSTM classifier. It includes:

- data integration,
- target construction,
- text cleaning,
- near-duplicate detection,
- grouping,
- word vectors,
- neural classification,
- multiple metrics,
- local attribution,
- topic modeling,
- manual analysis,
- an LLM comparison.

The dissertation therefore examines an end-to-end process of discovery from public text to explainable company metrics.

### 6.2 Multiple Companies and Metrics

The shared architecture is applied to revenue, EPS, vehicle counts, and search-engine market share. This is valuable because a method that works for only one metric is less general.

The old codebase does not yet establish clean cross-company generalization. It does create a useful comparative experimental matrix.

### 6.3 Binary and Four-Level Targets

Separating direction from magnitude is scientifically sensible:

- Binary asks: decrease or increase?
- Four-level asks: how large is the change?

The four classes have a natural order. The old model treats them as nominal classes; later ordinal models can explicitly use this structure.

### 6.4 Multi-Scale Aggregation

Group sizes of 5, 10, and 20 are more than hyperparameters. They represent a research question:

> How much collective public discourse is required for weak individual texts to form a stable signal?

Because of pseudoreplication, the old evaluation does not answer this question reliably. The idea of a prespecified evidence-budget ablation nevertheless remains strong.

### 6.5 The Temporal Fingerprint as a Finding in Its Own Right

Perhaps the most important scientific insight from the old high scores is not the claimed future prediction but a strong **temporal and regime fingerprint** in the language.

Word choice can encode reporting periods through:

- events,
- product cycles,
- campaigns,
- news sources,
- market sentiment,
- seasonal discussions.

This motivates research questions of its own:

- Which topics distinguish quarters?
- Which language patterns recur seasonally?
- Which signals appear before an earnings release?
- Which appear only afterward?
- What textual contribution remains after controlling for season and source?

### 6.6 A Shared Space for Prediction and Topics

Top2Vec supplies both word vectors for the LSTM and topics. This makes it possible to investigate whether the same semantic dimensions matter for both classification and interpretation.

This connection is original, but it must be trained cleanly within each fold or treated as a fixed external representation.

### 6.7 Token -> Tweet -> Topic

Combining Integrated Gradients, the original word, a POS tag, and the document topic is a sensible attempt to translate local model decisions into higher-level social-scientific interpretation.

Today, this path should be implemented as follows:

1. Explain only genuine future test cases.
2. Exclude PAD and SEP tokens.
3. Store signed and absolute attribution separately.
4. Aggregate correctly per tweet.
5. Only then summarize by topic.
6. Test stability across folds and seeds.

### 6.8 Multiple Topic Models and Quality Dimensions

A shared extractor interface supports Top2Vec and BERTopic. Topic quality is assessed through coherence, diversity, and silhouette.

This is scientifically better than assuming that one topic decomposition is the truth. Temporal stability and held-out generalization should also be measured.

### 6.9 Human–Machine and LLM Comparison

`ManualTopicAnalyzer` and `LLMTopicsCompare` compare manually or LLM-generated terms with model topics, both directly and in embedding space.

This is interesting as a triangulation design. A reliable study requires:

- blinded raters,
- predefined prompts,
- fixed similarity thresholds,
- inter-rater reliability,
- the same held-out documents for all systems.

### 6.10 Sound Data and Evaluation Ideas

Other elements worth preserving include:

- exactly one financial row per time interval,
- SimHash-based near-duplicate detection,
- class weights and an `EqualClassSampler`,
- precision, recall, F1, accuracy, and MCC,
- stored test indices,
- multiple temporal split variants,
- TensorBoard and checkpointing.

These elements demonstrate awareness of methodological problems. They do not automatically prove that every resulting run was valid.

---

## 7. Weak and Faulty Parts of `main`

### 7.1 Evaluation Design

| Problem | Consequence |
| --- | --- |
| Groups rather than company-quarters are evaluated | One quarterly target is counted many times. |
| Other groups from the same quarter can be in training and test | Period recognition becomes possible. |
| The primary K-fold run trains on all other blocks | Later texts can be in training for early test cases. |
| Validation is randomly stratified within the training pool | Training and validation can share the same periods. |
| The primary run lacks a seasonal baseline | Text skill and quarterly seasonality are conflated. |
| Some balancing occurs before the split | Temporal coverage and class frequencies are altered. |

### 7.2 Transductive Use of Top2Vec

Top2Vec is trained on the entire text corpus before the forecast split is fixed. Later test texts therefore affect:

- the vocabulary,
- semantic neighborhoods,
- the LSTM's initial vectors,
- the topic structure.

For a strict future test, the topic/embedding model must be trained only on past texts or declared as a clearly external, temporally fixed resource.

### 7.3 Target and Metric Definitions

- The connector returns the value from the same interval, while parts of the old README described Q+1.
- Ratio and percentage values have similar names.
- The four-class intervals have overlaps and a small gap.
- Tesla production, deliveries, and sales must not be treated as equivalent without documented provenance.

### 7.4 Padding and the Last Hidden State

Batches are padded to the longest sequence, but the LSTM receives neither true lengths nor a mask. The last hidden state can therefore contain a large amount of PAD processing.

The correct statement is not that a final hidden state is always wrong. It is problematic in this particular unmasked combination.

Possible repairs include:

- setting `padding_idx` in the embedding,
- using `pack_padded_sequence`,
- masked mean pooling,
- attention with a padding mask,
- hierarchical token/tweet/quarter aggregation.

### 7.5 Checkpoint Risk

When filenames already exist, Lightning can create versioned files such as `model-v1.ckpt`. Some scripts then manually reconstruct the unversioned path. This can load an old checkpoint.

The trainer itself tests with `ckpt_path='best'`. The blanket claim that every script necessarily evaluates an old checkpoint is therefore too broad. The risk in the manual reload paths is nevertheless real.

### 7.6 Date Conversion Error

The training scripts use `pd.to_datetime(post_date)` without `unit='s'`. Epoch seconds are consequently interpreted as nanoseconds and placed in 1970.

The numerical order may remain intact, but calendar year and quarter are wrong. The earlier diagnosis is correct here.

### 7.7 Reproducibility and Portability

- Seeds are not set consistently for Torch and NumPy.
- `loadModel` is partly hard-coded to `cuda:0`.
- `map_location` is missing during loading.
- `strict=False` can conceal incompatible state-dictionary keys.
- With ten epochs and a patience of ten, early stopping can barely stop early.

### 7.8 Errors in the Old Topic and Important-Word Path

| Finding | Meaning |
| --- | --- |
| `extractMostImportantWords.py` uses `df.head(50000)` rather than the stored test indices | Training texts can enter the explanation. |
| Integrated Gradients is divided by its own maximum for each sample | Rankings within a sample remain, but magnitudes become incomparable across samples. |
| There is no protection against division by zero | Zero attribution can produce NaN. |
| A group total is repeated for multiple tweets during flattening | Tweet-level values are aggregated incorrectly. |
| The Captum convergence delta is discarded | Attribution quality is not checked. |
| `findMostImportantTopicTweets.py` refers to an undefined variable | The script can fail on this path. |

The research question remains relevant. The old concrete output cannot, however, be described categorically as mechanically correct.

---

## 8. Review of the Earlier Automated Analysis

### 8.1 Correct or Correct in Substance

| Claim | Assessment | Reasoning |
| --- | --- | --- |
| The target is constant within a quarter. | Correct | All texts in a company-quarter share the same target realization. |
| Quarter-sharing splits permit period recognition. | Correct | Groups rather than quarters are separated. |
| The primary evaluation uses later blocks in training. | Correct | Test block `k`, training on all other blocks. |
| A seasonal baseline is necessary. | Correct | Earlier instances of the same calendar quarter carry a strong signal. |
| The date is interpreted as nanoseconds. | Correct | `unit='s'` is missing. |
| Fixed checkpoint names can evaluate an older run. | Correct for certain paths | Manual unversioned reload after Lightning checkpointing. |
| The data do not prove that tweets contain no financial information. | Correct | The diagnosis concerns the evaluation, not whether a signal exists. |

### 8.2 Partly Correct, but Overstated

| Claim | Correction |
| --- | --- |
| “20 distinct labels” | There are 20 quarterly outcomes, but only two or four class values. |
| “The LSTM does not learn in any configuration” | Certain runs collapsed; not every conceivable LSTM configuration was tested. |
| “The last hidden state is the cause” | The concrete issue is primarily unmasked padding combined with the final state. |
| “Every training script loads an old checkpoint” | Four manual reload paths are at risk; the trainer uses `best`. |
| “91.8% quarter recognition” | A methodologically plausible historical finding, but not exactly reproduced here without a committed result. |
| “All out-of-period protocols show no positive association” | Reported in the old analysis, but not reverified for all datasets using local raw artifacts. |

### 8.3 Wrong, Unsupported, or Not Required for This Project

| Claim | Assessment |
| --- | --- |
| The interpretation pipeline is mechanically sound. | Wrong; several concrete code errors contradict it. |
| The published 0.87/0.77 results could never have come from text. | Not provable; without the artifact, the exact mechanism of the historical checkpoint remains unknown. |
| Obtaining more than 20 targets requires a non-quarterly target. | Unnecessary; more years, companies, or external holdouts also increase the number of independent cases. |
| The target must be shifted to Q+1. | Only for a literal next-quarter forecast, not for a current-quarter nowcast. |
| The negative forecast diagnosis automatically makes the old topics correct. | Wrong; the interpretation path must be repaired separately. |

### 8.4 Overall Assessment of the Earlier Analysis

The main criticism is not hallucinated: the split and target structure create a real shortcut. What goes too far are absolute statements about all LSTMs, all training paths, and the entire interpretation system.

The precise assessment is:

> A strong diagnosis of the evaluation design, but overbroad conclusions about the architecture, experiments, and code quality.

---

## 9. The Current Quarterly Model

### 9.1 Target and Data

The target remains the four-level change in a quarterly business metric. Amazon, Apple, and Tesla are included.

The following are not used as current input features:

- the finance CSV,
- the current quarterly value,
- the current percentage target change,
- word embeddings,
- external data.

Past target classes are used for model training and the seasonal prior.

### 9.2 Target-Relevant Text Selection

A text enters the numerical aggregation if it contains both a company marker and a metric marker.

Examples:

| Company | Company marker | Metric marker |
| --- | --- | --- |
| Amazon | Amazon, AMZN | revenue, net sales, AWS sales |
| Apple | Apple, AAPL | EPS, earnings per share |
| Tesla | Tesla, TSLA | deliveries, delivery, production |

The result does not store complete texts. It stores only aggregated features.

### 9.3 Six Text Views

| View | Content | Hypothesis |
| --- | --- | --- |
| `all` | All relevant texts in the quarter | Base level |
| `late_third` | Final third | Later information is closer to the period end |
| `reported` | reported, actual, announced | Already reported or retrospective status |
| `forward_estimate` | estimate, consensus, guidance, future | Expectation language |
| `early_reported` | Reporting language in the first third | Proxy for the old reference level |
| `late_forward_estimate` | Estimates in the final third | Proxy for the expected new level |

### 9.4 Feature Families

The following are among the values computed from each view:

- total and relevant-text counts,
- proportion of relevant texts,
- number of percentage mentions,
- positive and negative percentages,
- median and quartiles,
- direct distribution of percentage values over four classes,
- proportions of reported, estimate, guidance, beat, miss, and future language,
- robust absolute metric levels,
- differences between early, late, reported, and expected levels.

### 9.5 Synthetic Example

Suppose early texts mention a reported delivery level of 70,000 units. Late estimates mention 81,000.

```text
estimated_change = (81,000 / 70,000 - 1) x 100 = 15.7%
```

15.7% falls into class 2. This is feature engineering from text, not retrieval of the true test target.

### 9.6 Numerical Text Classifier

The quarterly features are:

1. standardized using training quarters only,
2. clipped to the range `[-8, 8]`,
3. classified with regularized logistic regression.

Validation selects:

- the regularization,
- current features alone or combined with temporal differences,
- optional company identity.

### 9.7 Seasonal Prior

For a new Q2, the seasonal prior considers only earlier Q2 labels from the same company and uses them to produce a smoothed class distribution.

It does not use financial values as input, but it does use historical targets. It is therefore not a text feature.

### 9.8 Fusion

The general hybrid combines seasonal and text probabilities with a fixed weight.

For Tesla, a forward-level signal is added. It derives an expected change from late estimate levels and early reported levels.

### 9.9 Tesla Conflict Gate

The gate recognizes two specific conflict patterns between the base model and the numerical text model. In those cases, it replaces the prediction with the numerical text distribution.

The gate improves accuracy from 75.00% to 80.56%. Its thresholds were designed after inspecting the errors from 2017–2019, however. It is therefore exploratory rather than confirmatory.

### 9.10 Why CUDA Is Not Required

The current best model is not an LSTM. Regex aggregation, scaling, and logistic regression run on the CPU. CUDA was relevant to the earlier neural models, not the current 80.56% run.

---

## 10. Training and Future Testing

### 10.1 Rolling-Origin Design

| Test year | Training | Validation | Test |
| ---: | --- | --- | --- |
| 2017 | 2015 | 2016 | 2017 |
| 2018 | 2015–2016 | 2017 | 2018 |
| 2019 | 2015–2017 | 2018 | 2019 |

After selection, the model is refitted on training plus validation and evaluated on exactly the following test year.

Three seeds are used:

- 1337,
- 101337,
- 201337.

The probabilities from the three runs are averaged for each company-quarter.

### 10.2 What Does Not Leak in the Current Protocol

- No test label enters fitting, scaling, or selection.
- Training years are globally earlier than the validation year.
- The validation year is globally earlier than the test year.
- Each company-quarter combination is counted exactly once.
- Topic and word models are fitted only on earlier quarters.

### 10.3 Remaining Limitations

- Text from the entire quarter being evaluated is aggregated.
- A real-time cutoff at 25%, 50%, or 75% of the quarter is not yet a primary test.
- There are only 36 independent test cases.
- The Tesla gate is post hoc.
- Amazon and Apple are strongly supported by seasonal patterns.
- The local 2020 corpus is not dense enough for a comparable complete new holdout.

---

## 11. Results and Statistical Interpretation

### 11.1 Four-Class Metrics

| Model | Accuracy | MCC | Log loss | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Numerical text alone | 0.5000 | 0.3224 | 1.3078 | Pure numerical text signal |
| Seasonal prior without financial features | 0.6111 | 0.4743 | 0.9519 | Earlier instances of the same calendar quarter only |
| Seasonal + numerical text, fixed 50/50 | 0.6944 | 0.5854 | 1.0298 | General hybrid |
| Seasonal + Tesla forward | 0.7500 | 0.6633 | 0.9460 | Transparent variant without conflict gate |
| Seasonal + Tesla conflict gate | **0.8056** | **0.7387** | 0.9173 | Primary exploratory result |
| Primary bundle shuffle | 0.6944 | 0.5850 | 1.1094 | Text bundle shifted within the company |

### 11.2 Results by Company

| Company | Correct | Accuracy | MCC |
| --- | ---: | ---: | ---: |
| Amazon | 10 / 12 | 0.8333 | 0.7828 |
| Apple | 11 / 12 | 0.9167 | 0.8765 |
| Tesla | 8 / 12 | 0.6667 | 0.5664 |
| Overall | 29 / 36 | 0.8056 | 0.7387 |

### 11.3 Directional Diagnostic

If the same probabilities are collapsed to decrease versus increase, the results are:

- accuracy 0.9167,
- MCC 0.8003.

This is a simpler target and must not replace the primary four-class evaluation.

### 11.4 Uncertainty

For 29 correct decisions out of 36 cases, the Wilson 95% interval for accuracy is approximately:

```text
0.650 to 0.902
```

The interval is wide. True performance could be substantially below or above the point estimate.

### 11.5 Comparison with the Shuffle Control

The primary model is correct in four quarters in which the bundle-shuffle control is wrong. The reverse never occurs.

The paired, two-sided exact test gives:

```text
p = 0.125
```

This is a positive direction, but not a statistically significant difference at the 5% level.

### 11.6 What May Be Claimed

- A leakage-aware no-finance hybrid reaches an exploratory 80.56% accuracy and MCC 0.7387.
- Numerical text and expectation features help with certain Tesla decisions.
- The isolated numerical text branch contains a positive but limited signal.
- Amazon and Apple have strong seasonal target patterns.

### 11.7 What May Not Be Claimed

- Not that a pure-text model reaches 80%–90%.
- Not that the additional text contribution is statistically confirmed.
- Not that 2017–2019 remained an untouched final holdout after the gate was developed.
- Not that the model already forecasts the next quarter, Q+1.
- Not that topics cause a financial change.

### 11.8 The Next Confirmatory Test

The regexes, features, hyperparameters, fusion weights, and gate thresholds must be frozen. A completely new holdout is then required.

Only such a test can distinguish genuine generalization from retrospective adaptation.

---

## 12. Topics and Important Words

### 12.1 Why Integrated Gradients Is Not the Right Primary Method Here

The current numerical text branch consists of aggregated regex features and logistic regression. It has no token-embedding layer to which Integrated Gradients could be meaningfully applied.

For this branch, the exact linear attribution is:

```text
feature contribution = standardized feature value x class coefficient
```

The sum of these contributions plus the intercept reconstructs the decision score of the numerical text model.

### 12.2 Four Levels of Explanation

| Level | Computation | Interpretation |
| --- | --- | --- |
| Exact text-feature attribution | Standardized value times OVR coefficient | Exact for the numerical text branch |
| Model-adjacent cue words | Feature contributions are mapped to language families and observed cues | Feature family is exact; allocation to individual words is descriptive |
| Past-only important words | Quarter-stable class log odds from earlier quarters only | Temporally valid class association, not causal |
| Past-only topics | TF-IDF plus NMF on earlier quarters only; test texts are transformed | Context description, not additive model attribution |

### 12.3 Model-Adjacent Cues

Feature families are connected to concrete text patterns:

- reported,
- estimate,
- guidance,
- beat,
- miss,
- future,
- positive and negative direction,
- percentage markers,
- absolute-number markers,
- company and metric terms.

For medians, quantiles, and quarterly aggregation, assigning the value to individual words is no longer mathematically exact. It is therefore explicitly described as a descriptive bridge.

### 12.4 Past-Only Important Words

The word lexicon learns which terms were stably associated with a class in earlier quarters.

The order matters:

1. Only training plus validation data are used.
2. Words must recur across multiple quarters.
3. Only then is it checked which of those words occur in the future quarter.
4. The test label is not used to fit the word model.

This makes a one-off term from a single quarter less likely to become an allegedly important word.

### 12.5 Past-Only NMF Topics

For each company and test year, a small topic model is trained on the quarters available up to that point.

It uses:

- TF-IDF for text representation,
- NMF for topic decomposition,
- no more than 250 relevant documents per quarter,
- deterministic, quarter-balanced selection.

The future quarter is only projected into the already learned topic model.

### 12.6 Complete Decision Path

Each explanation keeps the following separate:

- seasonal probabilities,
- numerical text probabilities,
- forward-level probabilities,
- probabilities before the conflict gate,
- final probabilities,
- gate activation share,
- exact text-feature contributions,
- cue words,
- past-only important words,
- past-only topics.

This makes it visible whether a topic merely describes context or whether the text branch actually affected the final class.

### 12.7 Reproduction Safeguard

The explanation script replays the stored fold decisions. It stops if accuracy, MCC, or final classes do not match the primary result.

The successful replay reproduced:

- hybrid accuracy 0.8056,
- hybrid MCC 0.7387,
- numerical-text accuracy 0.5000,
- numerical-text MCC 0.3224.

---

## 13. Concrete Explanation Examples

All examples contain only aggregated terms and probabilities, not complete original texts.

### 13.1 Amazon 2017Q1

| Quantity | Result |
| --- | --- |
| True class | 3 |
| Numerical text class | 3 |
| Final class | 3 |
| Seasonal probability for class 3 | 0.750 |
| Numerical text probability for class 3 | 0.379 |
| Final probability for class 3 | 0.564 |

The strong seasonal path is confirmed by the text, not replaced by it.

Largest positive text-feature contributions for class 3:

- `all__miss_tweet_fraction`: +0.206,
- `forward_estimate__miss_tweet_fraction`: +0.192,
- `early_reported__log_percent_mentions`: +0.137.

Largest negative contributions:

- `reported__signed_percent_negative_fraction`: -0.153,
- `reported__percent_class_0_fraction`: -0.153,
- `late_third__miss_tweet_fraction`: -0.142.

Model-adjacent markers and terms include:

- `percentage_value`,
- `numeric_value`,
- actual,
- misses,
- estimates,
- revenue,
- miss.

The past-only important words for the final class begin with:

- business,
- misses,
- miss,
- earnings,
- sales.

Dominant topic areas include:

- revenue, Amazon, AWS, cloud,
- growth, revenue growth, year-over-year.

### 13.2 Tesla 2018Q1

Here, the conflict gate changes the decision.

| Branch | Probabilities `[0, 1, 2, 3]` | Argmax |
| --- | --- | ---: |
| Seasonal prior | `[0.313; 0.563; 0.063; 0.063]` | 1 |
| Numerical text | `[0.143; 0.456; 0.208; 0.193]` | 1 |
| Forward level | `[0.042; 0.042; 0.042; 0.875]` | 3 |
| Before conflict gate | `[0.135; 0.275; 0.088; 0.501]` | 3 |
| Final after gate | `[0.143; 0.456; 0.208; 0.193]` | 1, correct |

Largest exact contributions to numerical text class 1:

- `late_forward_estimate__metric_tweet_fraction`: +0.491,
- `all__log_percent_mentions`: +0.243,
- `early_reported__metric_tweet_fraction`: +0.228,
- `reported__metric_tweet_fraction`: +0.216.

After the abstract percentage and number markers, terms include:

- production,
- report,
- estimates,
- expect,
- miss.

The past-only important words begin with:

- model,
- model production,
- cars,
- quarter.

The dominant topics concern production, Model production, and deliveries.

This example shows both the benefit and the risk: the text branch correctly repairs the forward fusion. The decision, however, is made through a Tesla gate that was developed afterward. Topic words explain the context; they do not validate the gate thresholds.

---

## 14. Claim Ladder

The claim ladder separates technical feasibility, empirical observation, hypothesis, and unjustified conclusion.

| Claim | Status | Scientifically accurate formulation |
| --- | --- | --- |
| The complete pipeline is technically feasible. | Established | Local texts can be linked to reporting periods, grouped, classified, and traced back to words and topics. |
| Text groups contain strong class correlations. | Established under the old split | Under the earlier grouping and split conditions, the target class is highly separable. |
| Language encodes time, events, and market regimes. | Strong hypothesis | The old high scores and the seasonal and shuffle diagnostics motivate a temporal-fingerprint test. |
| Certain words and topics are interpretively relevant. | Candidates | Stability must be remeasured on past-only fits and future tests. |
| Text predicts unknown later quarters with 87% accuracy. | Not established | The old 0.87 must not be interpreted as strict future generalization. |
| The current pure-text model reaches 80.56%. | False | 80.56% belongs to the hybrid; numerical text alone reaches 50.00%. |
| A topic causes a quarterly change. | Not established | Attribution and topic assignment show model association, not economic causation. |

### The Dissertation Contribution Worth Preserving

The scientific core consists of four elements:

1. The design of a modular information system connecting social-media text with company metrics.
2. Empirical evidence of strong temporal and event-related language signatures.
3. Multi-scale aggregation as a response to weak individual tweets.
4. Multi-level interpretation from token attribution to topic and LLM comparison.

These contributions remain even though the old forecast accuracy must be reinterpreted.

---

## 15. Privacy and Tweet-Content Audit

### 15.1 Result

The audit of the visible branch tree is a **FAIL** under a strict zero-tweet-content requirement.

There is no large committed raw dataset. Some files nevertheless contain complete or tweet-like content.

### 15.2 High Priority

| File | Finding | Recommended action |
| --- | --- | --- |
| `tweetsCompanyNumbersPrediction/src/tests/companyTweetsDummy.csv` | Several complete text rows with handles or links | Replace with deterministic synthetic texts |
| `tweetsCompanyNumbersPrediction/src/predictSingleTweetGroup.py` | Hard-coded longer tweet group and commented examples | Synthesize or load through the CLI |
| `tweetsCompanyNumbersPrediction/src/tests/TestNearDuplicateDetector.py` | Realistic-looking market posts | Rewrite as neutral synthetic content |
| `tweetsCompanyNumbersPrediction/src/tests/TweetSentimentAnalysisTest.py` | Apple/AAPL-like post texts | Rewrite as neutral synthetic content |

### 15.3 Lower Priority

Clearly synthetic text fixtures can be found in, among others:

- `PipelineTest.py`,
- `TweetTextFilterTransformerTest.py`,
- `nlpvectorstest.py`,
- `HyperlinkRemoverTest.py`.

They are not bulk raw data. Under an absolutely strict zero-content policy, however, even these examples should avoid handles, real links, and market-like wording.

### 15.4 What Is Already Clean

- No large aggregate CompanyTweets dataset in the repository tree.
- No committed trained checkpoints or exported tweet groups.
- `tokenizer.json` contains a vocabulary, not coherent posts.
- The new result JSON files contain metrics, features, terms, and topic aggregates, but no complete texts.
- `numeric_text_topics_important_words.json` contains no authors, handles, URLs, or tweet IDs.

### 15.5 Cleanup Plan

1. Replace `companyTweetsDummy.csv` with clearly synthetic artificial sentences.
2. Remove hard-coded demo texts from `predictSingleTweetGroup.py` or load them from input.
3. Rewrite the near-duplicate and sentiment fixtures in neutral language.
4. Add a pre-commit scanner for `body` CSV schemas, `t.co`, handles, cashtags, and long text literals.
5. Rerun all tests.
6. If required for publication, audit the Git history separately for historical blobs. Cleaning the current tree does not remove old commits.

### 15.6 Minimum Release Standard

- No high-risk hit in the branch-tree scan.
- No CSV or JSON file with multiple complete post-like sentences.
- No authors, handles, tweet URLs, or platform IDs in fixtures.
- Research artifacts store only aggregations and metrics.

---

## 16. Recommended Research Roadmap

### Priority P0

#### Freeze the Gate and Test It on New Data

The Tesla conflict gate must not be modified further using 2017–2019. The architecture and thresholds must be frozen and tested on new quarters.

#### Always Report Baselines Together

Every run should include at least:

- a global majority baseline,
- a seasonal baseline,
- a persistence baseline,
- the pure text branch,
- the hybrid,
- a text-shuffle control.

#### Use the Company-Quarter as the Primary Unit

No primary metric may count individual tweets or tweet groups as independent financial events.

### Priority P1

#### Early Nowcast Cutoffs

Features should be evaluated at fixed fractions of the quarter:

- 25%,
- 50%,
- 75%,
- 100%.

This reveals whether the signal is available early or arises only from retrospective reporting and summary texts.

#### Hierarchical Company Model

A shared model can learn general language patterns while also using company-specific coefficients or adapters.

#### Interpretation Stability

Important words and topics should be compared across seeds, folds, and companies. A term that appears in only one run is a weak finding.

#### Target Provenance

The following should be documented for each company:

- data source,
- metric name,
- unit,
- reporting date,
- reporting periods,
- definition of the percentage change,
- class boundaries.

### Priority P2

#### Fair LSTM Comparison

A new LSTM or attention branch should:

- derive its vocabulary and trainable representation only from past periods,
- mask or pack correctly,
- set `padding_idx`,
- model token -> tweet -> quarter hierarchically,
- report the text contribution separately,
- compete against identical seasonal and shuffle controls.

#### LLM Comparison

The LSTM, topic model, and LLM should receive the same held-out company-quarters. Prompts, temperature settings, and evaluation must be defined in advance.

### The Target Remains a Quarterly Metric

Obtaining more independent data does not require switching to a daily stock price. Suitable approaches include:

- more completed quarters,
- additional, clearly defined companies,
- an external holdout,
- multiple metrics with clear provenance.

---

## 17. Reproduction and Code Evidence

### 17.1 Run the Current Numerical Quarterly Model

From the `tweetsCompanyNumbersPrediction/src` directory:

```bash
python trainNumericTextSignalQuarterModel.py
```

Output:

```text
output/numeric_text_signal_quarter_results.json
```

### 17.2 Generate Topics and Important Words

```bash
python extractNumericQuarterTopicsAndImportantWords.py
```

Output:

```text
output/numeric_text_topics_important_words.json
```

The run processed a total of:

- 5,170 relevant Amazon texts,
- 4,271 relevant Apple texts,
- 34,657 relevant Tesla texts,
- 20 available quarters for each company.

The quarter-balanced cap of 250 documents applies to the topic and word models. The prediction features themselves remain unchanged.

### 17.3 Tests

```bash
python -m unittest tests.alltestsuite
```

The most recent complete run passed:

- 111 registered tests,
- 19 additional tests.

### 17.4 Important Current Files

| Subject | File |
| --- | --- |
| Numerical text features | [`NumericQuarterTextFeatures.py`](tweetsCompanyNumbersPrediction/src/classifier/NumericQuarterTextFeatures.py) |
| Training, seasonality, and Tesla gate | [`trainNumericTextSignalQuarterModel.py`](tweetsCompanyNumbersPrediction/src/trainNumericTextSignalQuarterModel.py) |
| Exact attribution and past-only topics | [`NumericQuarterTextExplanations.py`](tweetsCompanyNumbersPrediction/src/featureinterpretation/NumericQuarterTextExplanations.py) |
| Replay and explanation output | [`extractNumericQuarterTopicsAndImportantWords.py`](tweetsCompanyNumbersPrediction/src/extractNumericQuarterTopicsAndImportantWords.py) |
| Explanation tests | [`NumericQuarterTextExplanationsTest.py`](tweetsCompanyNumbersPrediction/src/tests/NumericQuarterTextExplanationsTest.py) |

### 17.5 Central `main` Code Evidence

The line references refer to `main` commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6`.

| Subject | File and lines |
| --- | --- |
| Same-period join | `tweetnumbersconnector/tweetnumbersconnector.py:22-48` |
| Per-class groups | `nlpvectors/DataframeSplitter.py:40-85` |
| Tweet group and SEP | `nlpvectors/TweetGroup.py:27-50` |
| LSTM final state | `classifier/LSTMNN.py:9-30` |
| Primary K-fold | `trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py:71-117` |
| Global 80/20 split | `trainNumbersPredictionModelTemporalSplit.py:60-78` |
| Expanding window per class | `trainNumbersPredictionModelStratifiedExpandingWindowPerClass.py:76-113` |
| Trainer and checkpointing | `classifier/Trainer.py:20-39` |
| Metrics | `classifier/ClassificationMetrics.py:12-34` |
| Topic backends | `topicmodelling/TopicExtractor.py:31-179` |
| Topic quality | `topicmodelling/TopicEvaluation.py:24-48` |
| Word, POS, and topic mapping | `featureinterpretation/InterpretationDataframeUtil.py:8-31` |
| LLM topic comparison | `topicmodelling/llmcomparison/LLMTopicsCompare.py:43-99` |
| Near-duplicate detection | `tweetpreprocess/nearduplicates/NearDuplicateDetector.py:34-66` |

### 17.6 New Result Artifacts and Privacy

The new explanation file stores:

- targets and predictions at the company-quarter level,
- probabilities for each model branch,
- feature values and coefficient contributions,
- individual aggregated cue terms,
- past-only important words,
- topic words and topic weights.

It does not store:

- complete texts,
- authors,
- handles,
- URLs,
- tweet IDs,
- document IDs.

A recursive key audit blocks typical raw-text and identifier fields.

---

## 18. Final Assessment

### The Old Implementation

The old `main` codebase is scientifically interesting because it:

- implements a complete information system,
- covers multiple companies and metrics,
- aggregates weak individual texts,
- connects prediction with interpretation,
- evaluates topic quality quantitatively,
- combines manual, neural, and LLM perspectives,
- exposes a strong temporal fingerprint in the language.

It is not, however, convincing evidence of 87% genuine future forecasting. The main reasons are quarterly pseudoreplication, training on later blocks, the missing seasonal baseline, transductive embeddings, and concrete code problems.

### The Earlier Automated Analysis

The earlier automated analysis correctly identified the central weakness in the evaluation design. It became unreliable where it generalized from a justified finding to universal claims about all LSTMs, all checkpoints, or the entire interpretation pipeline.

### The Current Model

The current system substantially improves temporal evaluation and reaches an exploratory 80.56% accuracy and MCC 0.7387. This result belongs to a hybrid. The pure numerical text branch reaches 50.00% accuracy and MCC 0.3224.

The Tesla conflict gate is post hoc. Therefore, 75.00% accuracy and MCC 0.6633 remain the more transparent ungated reference until new quarters confirm the gate.

### Topics and Important Words

The topic and word analysis is now more closely connected to the actual decision path. Exact text-feature contributions, model-adjacent cues, past-only important words, and past-only topics are reported separately.

The most important limitation remains:

> Topics describe a model's context. They establish neither a causal effect on the financial metric nor an additive contribution to the seasonal prior or the Tesla gate.

### Final Conclusion

The project has two simultaneously true findings:

1. The old high accuracy must not be interpreted as a clean future forecast.
2. The old research platform, the temporal language signatures, and the connection between prediction and interpretable topics are scientifically valuable and deserve a leakage-aware continuation.
