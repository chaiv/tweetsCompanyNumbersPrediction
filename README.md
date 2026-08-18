# Predicting Company Financial Metrics from Tweets

Research code accompanying the PhD thesis *"Machine learning as a tool for developing an information system for predicting company metrics from social media data"* by Vitali Chaiko, written at [UniBIT](https://www.unibit.bg/en) in cooperation with [Vienna International Studies](https://viennastudies.com/home) and [AfW Bad Harzburg](https://www.afwbadharzburg.de).

## Abstract

This research aims to improve organizational management and processes by designing an AI-powered automated system for the prediction of various organizational financial metrics based on machine learning and social media data. In addition to predictive functionality, the system is designed to extract the most important social and economic topics and keywords that contribute to the prediction performance. The concept of the research follows an exploratory step-by-step approach, resulting in a quantitative evaluation.

The study is based on three datasets:

1. Publicly accessible **Twitter data** for five leading NASDAQ-listed U.S. companies (Amazon, Apple, Google, Microsoft and Tesla) covering the years 2015 to 2020.
2. Public **financial metrics** associated with these companies, such as quarterly revenue.
3. Socially relevant **topics and social issue keywords** that were visible to the public between 2015 and 2020.

Custom neural networks and topic models were trained successfully, demonstrating the practicability and generalization of predictions on financial metrics based on text information, and achieving high evaluation metrics, including an **accuracy of 0.87** and a **Matthews correlation coefficient of 0.77**. The system further extracts meaningful insights, such as a connection between the Black Lives Matter movement and Amazon's Christmas campaign, or a Brexit-driven investor shift to cryptocurrency.

Directions for future research include improving the performance of topic models, which underperformed compared to the prediction models with a UCI coherence score of 0.44, and real-world testing of the system in context-specific use cases with domain experts.

## Publications & Resources

| Resource | Link |
| --- | --- |
| PhD thesis (PDF) | [EN - Dissertation - Vitali Chaiko](https://ras.nacid.bg/api/reg/FilesStorage?key=6db2fa58-4805-4803-9d11-3dc09e838e6e&mimeType=application/pdf&fileName=EN%20-%20Dissertation%20-%20Vitali%20Chaiko%20-%2010.03.2026.pdf&dbId=2) |
| Thesis bibliographic record | [COBISS 79716360](https://plus.cobiss.net/cobiss/bg/en/data/cobib/79716360) |
| Publication 1 | [Comparing ChatGPT And LSTM In Predicting Changes In Quarterly Financial Metrics](https://www.researchgate.net/publication/381910297_Comparing_ChatGPT_And_LSTM_In_Predicting_Changes_In_Quarterly_Financial_Metrics) |
| Publication 2 | [Buditeli proceedings 2024 (UNIBIT)](https://buditeli.unibit.bg/images/proceedings/2024/chaiko.pdf) |
| Publication 3 | [UNIBIT e-journal proceedings 2024, book 2](https://e-journal.unibit.bg/images/proceedings/2024/book2/5_Statiq_Vitali_Chaiko.pdf) |
| Primary dataset | [Tweets about the Top Companies from 2015 to 2020 (Kaggle)](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) |

---

## Method Overview

The system formulates the prediction task as supervised text classification over aggregated tweets:

1. **Tweets are joined with financial figures.** In the current prepared datasets, each tweet is labelled with the change of the reporting period that contains it (quarterly revenue, EPS, car sales or search engine market share). Because the figure is published after the period closes, this is a *nowcast* of the current quarter, not a forecast of the following quarter. A genuine next-quarter experiment must shift the target by one reporting period.
2. **The percentage change is discretized into classes.** Two schemes are used: a binary scheme (`decrease` / `increase`) and a four-class scheme (`decrease`, `weak`, `moderate`, `strong increase`).
3. **Tweets are aggregated into tweet groups** of N tweets sharing the same class, with N set to 5, 10 or 20. A group forms one training sample, since an individual tweet provides insufficient signal for the prediction task.
4. **An LSTM classifier** operating on Top2Vec word embeddings predicts the class of a tweet group.
5. **Integrated Gradients (Captum)** attributes the prediction back to individual tokens, and the **Top2Vec and BERTopic** models map these tokens to interpretable topics.

## Repository Structure

The implementation is located under [`tweetsCompanyNumbersPrediction/src`](tweetsCompanyNumbersPrediction/src). The `.py` files at the top level are executable scripts; the sub-packages contain the reusable library code.

### Configuration

| File | Purpose |
| --- | --- |
| [`PredictionModelPath.py`](tweetsCompanyNumbersPrediction/src/PredictionModelPath.py) | Central experiment registry. Each constant, for example `AMAZON_REVENUE_10_LSTM_MULTI_CLASS`, bundles the NASDAQ tag, dataset paths, word vector path, checkpoint directory, topic model path, tweet group size and class mapper of one experiment. **Every script selects its experiment by assigning one of these constants to `predictionModelPath`.** |
| [`TrainingMode.py`](tweetsCompanyNumbersPrediction/src/TrainingMode.py) | Enum of the four evaluation strategies (`SUBSEQUENT`, `STRATIFIED_TEMPORAL`, `TEMPORAL_SPLIT`, `STRATIFIED_KFOLD_TEMPORAL_PER_CLASS`) and whether they sort tweets temporally. |
| [`tweetpreprocess/DataDirHelper.py`](tweetsCompanyNumbersPrediction/src/tweetpreprocess/DataDirHelper.py) | Resolves the data root directory. **Adjust the paths here before running any script.** |

### Library packages

| Package | Contents |
| --- | --- |
| `tweetpreprocess` | Tweet querying and sorting, date to timestamp conversion, text and stop word filtering (`wordfiltering/`), near duplicate detection, the class calculators (`FiguresIncreaseDecreaseClassCalculator`, `FiguresMultiClassCalculator`, `FiguresPercentChangeCalculator`) and `EqualClassSampler` for class balancing. |
| `tweetnumbersconnector` | Joins each tweet with the financial figure of its reporting period (`TweetNumbersConnector`) and derives the target classes (`FinancialFiguresClassifier`). |
| `pipeline` | `FeatureDataframePipeline`, which composes the preprocessing steps into the labelled tweet dataframe. |
| `nlpvectors` | Tokenization (`TweetTokenizer`), vocabulary creation, word vector and ID encoders, `DataframeSplitter` for the construction of tweet groups, `TweetGroup` and the dataframe conversion helpers. |
| `classifier` | `LSTMNN` and `LSTMWithDropout` (PyTorch Lightning modules), `CreateClassifierModel` as model factory, `Trainer` (Lightning training loop with early stopping, checkpointing and TensorBoard logging), `TweetGroupDataset`, `ClassificationMetrics`, `PredictionClassMappers` (`BINARY_0_1`, `MULTICLASS_4`) and `transformer/Predictor` for batched inference. |
| `featureinterpretation` | `AttributionsCalculator` (Captum Integrated Gradients), `ImportantWordsStore`, `TokenScoresSort` and dataframe utilities for the analysis of the most important words. |
| `topicmodelling` | `TopicModelCreator` and `TopicExtractor` for Top2Vec and BERTopic, `TopicEvaluation` (coherence, diversity), `ManualTopicAnalyzer` and `llmcomparison/LLMTopicsCompare`. |
| `exploredata` | Descriptive statistics and plots for the tweet and financial dataframes, as well as POS tagging. |
| `sentiment` | VADER based sentiment analysis of tweets and words. |
| `tests` | Unit tests. [`tests/alltestsuite.py`](tweetsCompanyNumbersPrediction/src/tests/alltestsuite.py) aggregates the complete suite. |

---

## Reproducing the Experiments

### 1. Setup

```bash
git clone https://github.com/chaiv/tweetsCompanyNumbersPrediction.git
```

```bash
pip install -r tweetsCompanyNumbersPrediction/requirements.txt
```

The code targets Python 3.10 and assumes a CUDA-capable GPU, since `Trainer` defaults to `accelerator='cuda'`. Set it to `'cpu'` if no GPU is available. All scripts are executed from the `src` directory, which must also be on the `PYTHONPATH`:

```bash
cd tweetsCompanyNumbersPrediction/src
```

### 2. Prepare the data directory

Download the [Kaggle dataset](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) and merge `Tweet.csv` with `Company_Tweet.csv` into `CompanyTweets.csv` using [`tweetpreprocess/companyTweetMerge.py`](tweetsCompanyNumbersPrediction/src/tweetpreprocess/companyTweetMerge.py).

Place the result in a `companyTweets` folder inside the data root, together with a CSV of the financial metric per company, which requires the reporting date and a `percent_change` column. Then point `DataDirHelper` to that root. The file names expected by each experiment are listed in `PredictionModelPath.py`.

### 3. Build the labelled tweet dataframe

```bash
python createTweetsWithNumbers.py
```

This step joins the tweets with the financial figures and writes the labelled dataframe to `predictionModelPath.getDataframePath()`. The experiment is selected at the top of the script.

### 4. Train the topic model and export the word vectors

```bash
python trainTop2VecTopicModel.py
```

```bash
python top2VecWordVectorsToFile.py
```

The first script trains a Top2Vec model on the tweet corpus. The second script exports its vocabulary as word2vec-format vectors, extended by the `<PAD>`, `<UNK>` and `<SEP>` tokens, to `getWordVectorsPath()`. These vectors serve as the pretrained embeddings of the LSTM. Optionally, `createTokenizerLookup.py` builds the token to original word lookup required by the interpretation scripts, and `trainBertTopicModel.py` trains the BERTopic alternative.

### 5. Train the prediction model

The evaluation strategy is chosen according to the research question:

| Script | Split strategy |
| --- | --- |
| `trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py` | 10-fold stratified cross-validation with rotating temporal test blocks per class. This is the main evaluation. |
| `trainNumbersPredictionModelStratifiedExpandingWindowPerClass.py` | Expanding window per class, so that the training data always precedes the test data temporally and no information leaks from the future. |
| `trainNumbersPredictionModelTemporalSplit.py` | Strict 80/20 temporal split, which corresponds most closely to a real forecasting setting. |
| `trainNumbersPredictionModelStratifiedTemporalOrder.py` | Latest tweets as test set, stratified across classes. |
| `trainNumbersPredictionModelOnlySubsequentTweetsOrder.py` | Groups of N subsequent tweets without a temporal test set, used for the analysis of topics and important words. |

```bash
python trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py
```

Each fold writes a checkpoint `tweetpredict_fold{k}.ckpt` and its test indices `test_idx_fold{k}.npy` to `getModelPath()`, prints a classification report and the MCC, and the run concludes with the mean MCC across all folds. TensorBoard logs are written to `companyTweets/modellogs`.

For a leakage-resistant comparison of Top2Vec embeddings, a compact padding-safe BiLSTM and a
seasonal/text hybrid, run:

```bash
python trainQuarterAlignedEmbeddingModel.py --experiment apple-eps
```

By default this fits on 2015-2017, uses 2018 only to select the epoch count, refits on 2015-2018
and evaluates once on 2019. Tweet groups never cross a reporting-quarter boundary, every training
quarter contributes the same number of groups, and the output reports both group-level metrics and
one aggregated decision per independent quarter. The no-text majority and seasonal baselines are
printed beside every model result. Other choices include `tesla-sales`, `amazon-revenue-binary` and
`amazon-revenue-4class`.

To measure the temporal shortcut directly, `python evaluateQuarterRecognition.py` trains a TF-IDF
classifier to identify the exact reporting quarter of an Apple tweet group. This is a diagnostic of
period-specific vocabulary, not a financial prediction.

For the stronger multi-view ablation, run `python trainMultiViewQuarterModel.py --experiment
apple-eps`. It compares hierarchical Top2Vec tweet vectors, frozen MiniLM sentence embeddings,
leakage-conscious metadata, their fusion, and a fusion with an explicit seasonal prior. Final
likes, retweets and comment counts are excluded because their observation time is unknown.

For a pooled company-quarter experiment with substantially fewer duplicated targets, run
`python trainQuarterSequenceModel.py`. It encodes chronological bins of current-quarter tweets
with frozen MiniLM embeddings, processes them with a bidirectional LSTM, and separately processes
the four preceding quarterly financial observations with a second LSTM. The `calendar`, `text`,
`financial` and `fusion` ablations use only the locally configured tweet and financial CSV files.
The `fusion-shuffled-text` negative control keeps the same architecture but deliberately breaks
the text-to-quarter assignment within each company, so an apparent gain cannot automatically be
attributed to genuine quarterly text content.

The current quarter's financial value and percentage change are never model inputs: the target
remains its four-class quarterly change. Accuracy and MCC are reported after tweet-bag predictions
are averaged to one decision per independent company-quarter, over five seeds by default.

For a broader future-only result, `python trainRollingFutureQuarterModel.py` evaluates the
financial and text-fusion models with rolling test years 2017, 2018 and 2019. In every fold, all
training quarters precede the validation year and the validation year precedes the test year.
Metrics are aggregated over 36 independent future company-quarters and are also reported
separately for Amazon, Apple and Tesla.

For a text-focused, directly interpretable residual experiment, run:

```bash
python trainRelevantTextResidualModel.py --experiment amazon-revenue-4class
```

This pipeline first selects finance-related tweets without consulting the target: a keyword
prefilter is reranked by a frozen MiniLM encoder and balanced over eight chronological bins of the
quarter. The prediction network retains Top2Vec token IDs and uses a token-attention BiLSTM inside
each tweet followed by a tweet-attention BiLSTM. Its text logits are learned as a gated residual
over a strictly lagged financial/seasonal model, with modality dropout during training. Twelve
independent text bags are averaged to one prediction for each company-quarter. Rolling-origin test
years 2017, 2018 and 2019, three seeds, Accuracy and MCC, and a shuffled-quarter text control are
written to `output/relevant_text_residual_results.json`. Available experiments are
`amazon-revenue-4class`, `apple-eps` and `tesla-sales`; CUDA is selected automatically.

`featureinterpretation/HierarchicalQuarterAttributions.py` applies Layer Integrated Gradients to
the *scaled text contribution actually added to the fusion logits*. It preserves raw signed and
absolute values at token and tweet level and can aggregate held-out tweet contributions by topic.
This avoids the former per-group normalization, which made attribution magnitudes across groups
incomparable. Important-word extraction now also uses the saved held-out split rather than a fixed
prefix of the dataset.

The pooled text-change experiment is run with:

```bash
python trainTextDeltaOrdinalModel.py
```

It aggregates up to 512 finance-relevant tweets in each of eight chronological bins (rather than
training on a few duplicated tweet bags), and concatenates the current representation with its
change from the previous quarter and the same quarter of the previous year. A shared multi-company
GRU is pretrained text-only with two targets derived from the same quarterly CSV: three ordered
class thresholds and continuous percentage-change regression. Its logits are then normalized and
fused with the strictly lagged financial branch using an explicit 40% text weight that cannot be
learned down to zero. The script evaluates rolling future years over five seeds and reports finance,
text-only, fusion and a test-time shuffled-text control. `--text-weight 0.6` or `0.8` can be used for
the stronger ablation; the default 40% setting is deliberately not selected on the test results.

To test whether the correlations learned by the original Top2Vec LSTM contain reusable future
signal, run:

```bash
python trainPastOnlyTeacherStudentModel.py
```

For each rolling test year, this trains the original two-layer mean-pooled LSTM teacher only on
earlier labelled tweet groups. Its 256-dimensional pre-head representation is summarized per
quarter through class probabilities, confidence, hidden norms and cosine similarity to
past-quarter class prototypes. Current, previous-quarter and year-over-year text changes form one
quarter-level feature row; raw hidden coordinates from separately trained company teachers are
never mixed. A ridge student compares text-only prediction with a residual correction to the
strictly lagged same-quarter-last-year financial baseline. The output reports Accuracy, MCC and
MAE for finance, text, fusion and within-company shuffled text over the 36 independent 2017-2019
company-quarters.

The word vectors are frozen but were originally trained without labels on the full local tweet
corpus. The teacher's supervised parameters and all prototype/statistical features are past-only,
but this embedding initialization is therefore transductive. The target remains exclusively the
current quarter's financial percentage-change class; no extra target or external dataset is used.

The regularized enhanced variant is run with:

```bash
python trainEnhancedPastOnlyTeacherStudentModel.py
```

It adds a training-quarter identification head to the past-only financial-class teacher and a
zero-initialized low-rank adapter while keeping the Top2Vec table frozen. The quarter student
compares LSTM text, safe current-quarter metadata and both views together. Metadata comprises tweet
volume, unique-writer diversity/concentration, within-quarter timing, and URL/cashtag/number/text
ratios from the same local CSV; engagement counters are excluded. The immediately preceding
validation year selects 2, 4 or 8 past-only PCA components, ridge regularization and a residual gate
of 0%, 25%, 50% or 100%. A zero gate falls back exactly to the lagged financial baseline. Separate
controls shuffle only LSTM text or all current-quarter signal within each company. Results are
written to `output/enhanced_past_only_teacher_student_results.json` with Accuracy, MCC and MAE over
the rolling 2017-2019 company-quarters.

For a model that follows the original tweet-group idea but removes the financial model and word
embeddings completely, run:

```bash
python trainPureTextQuarterModel.py
```

Every training sample is again a group of ten tweets carrying the current quarter's financial
class, but groups never cross quarter boundaries and evaluated quarters are strictly later than
training and validation quarters. Four independent text views are fitted only on past groups:
word TF-IDF, character TF-IDF, class-specific important-word log odds, and a compact view of VADER
sentiment, forward/uncertain language, punctuation, numbers, URLs, cashtags, author diversity and
within-group timing. No financial value, financial lag, Top2Vec vector or neural teacher is read.
Validation selects regularization, important-word temperature and convex late-fusion weights.
Group probabilities are then averaged to exactly one prediction per company-quarter. The JSON
output includes Accuracy, MCC, a within-company shuffled-quarter control, and the top past-only
important words for every class, company and rolling fold so they can be used by the subsequent
topic/important-word analysis.

`python trainSemanticPureTextQuarterModel.py` is the anti-shortcut follow-up. It retains raw word
TF-IDF as a control but removes URLs/domains, usernames, cashtags, years, dates, month/day names and
tracking/newsletter boilerplate for the semantic view. A second view keeps only tweets with
target-independent business-event language such as revenue, earnings, sales, deliveries, demand,
margins, guidance or estimates. Its stable important-word view computes prevalence per past
quarter rather than per duplicated group and accepts a positive class word only when it recurs in
at least two past quarters of that class. Validation selects one complete view instead of fitting
many fusion weights to only twelve validation quarters. The target and rolling future evaluation
remain unchanged and no financial input or embedding is introduced.

`python trainQualityFilteredPureTextQuarterModel.py` rebuilds the ten-tweet samples themselves
before applying the semantic experiment. Only company or financial-event tweets are retained;
exact normalized duplicates and common stock-promotion templates are removed, and each author is
capped within each quarter before new chronological groups are formed. The output records retained
tweet fractions and available group counts for every company/quarter. This separates a failure due
to noisy generic Twitter traffic from a failure of genuinely company/metric-related language while
preserving the same past-only selection, refit and future-quarter evaluation.

`python trainTemporalAggregationPureTextModel.py` keeps the best finance-event text view but treats
group-to-quarter aggregation as a validation-only choice. Alongside the ordinary probability mean,
it evaluates group voting, geometric pooling, early or late half-quarter evidence, the last third
of the quarter and the most confident quarter of groups, with optional probability-temperature
scaling. The selected rule is refit on train plus validation groups before the future quarter is
scored. This tests whether the useful language is concentrated at a particular point in the target
quarter without exposing any future label during aggregation selection.

`python trainOrdinalPureTextQuarterModel.py` retains the same four quarterly-number classes but
learns them through three ordered sparse decisions: class greater than 0, greater than 1 and
greater than 2. Raw, anti-shortcut semantic and finance-event TF-IDF are selected only on the
preceding validation year. Its JSON reports the primary four-class Accuracy/MCC and, separately,
the decrease-versus-increase direction implied by the same probabilities. The direction result is
diagnostic and never replaces the four-class target.

`python trainPooledTextDeltaQuarterModel.py` removes group-label pseudoreplication by training on
one mean sparse-text row per independent company-quarter. Amazon, Apple and Tesla share a model;
current text can be concatenated with its sparse change from the previous quarter and the same
quarter one year earlier. Company identity is the only optional metadata. The experiment remains
free of financial values, financial baselines, embeddings and external data.

For target-context numbers mined directly from every local tweet, run:

```bash
python trainNumericTextSignalQuarterModel.py
```

This extracts signed percentages and robust absolute values only near Amazon revenue/net-sales,
Apple EPS, or Tesla delivery/production language. It separately aggregates the whole quarter, the
late third, reported language, forward estimates, early reports and late estimates. The rolling
model uses 2015 onward for training, the immediately preceding year for selection, and untouched
future test years 2017-2019. It does not read a financial CSV as an input model, use embeddings, or
use external data; the primary target is still the four-class quarterly-number change.

The transparent seasonal-plus-numeric model reaches 75.00% Accuracy / 0.663 MCC over 36 future
company-quarters. An additional Tesla conflict gate reaches 80.56% / 0.739, versus 69.44% / 0.585
when the complete numeric text bundle is shifted within each company. That final gate is explicitly
marked exploratory in the JSON because it was designed after inspecting the same 2017-2019
diagnostics. It is therefore evidence that the local text representation can reach the requested
range on this evaluation, not an independent confirmatory estimate. The local 2020 coverage has
only 6 Amazon, 17 Apple and 14 Tesla tweets in Q1 and cannot serve as a complete new holdout.
With 29 correct predictions out of 36, the Wilson 95% interval is approximately 65.0%-90.2%.
Against the fully shifted text bundle, four quarters improve and none worsen, but the paired exact
two-sided p-value is 0.125. The JSON records this statistical audit so the exploratory point
estimate cannot be mistaken for a precise or independently significant result.

The matching dissertation-step-3 explanation is generated with:

```bash
python extractNumericQuarterTopicsAndImportantWords.py
```

It replays every stored rolling-fold selection and first verifies that the 80.56% Accuracy and
0.739 MCC predictions are reproduced. For the selected numeric-text logistic branch it reports
exact signed feature contributions (`standardized value * OVR coefficient`). Those aggregate
features are connected to matched reporting, estimate, guidance, direction, metric and numeric
cues. A quarter-stable important-word lexicon and a TF-IDF/NMF topic model are fitted separately
for every rolling fold using only train plus validation quarters; they are then applied to the
future test quarter. Important words are therefore past-only class associations and topics are
contextual summaries, not falsely presented as causal contributions to the seasonal prior or the
exploratory Tesla gate. The output
`output/numeric_text_topics_important_words.json` contains terms, counts, feature values and topic
descriptors only - no raw tweet bodies, authors, handles, URLs or tweet IDs.

### 6. Evaluate

```bash
python calculateClassificationMetrics.py
```

The script reloads a checkpoint together with its saved test split and reports precision, recall, F1, accuracy and MCC. `TRAINING_MODE` must match the training script that produced the checkpoint, because it determines whether tweet groups are sorted temporally.

### 7. Extract the most important words and topics

```bash
python extractMostImportantWords.py
```

```bash
python calculateTopImportantWordsWithTopics.py
```

The first script computes the Integrated Gradients attributions per token for a given observed class and stores them as `importantWordsClass{N}{Company}.csv`. The second script ranks these attributions and adds the untokenized word, the POS tags and the matching Top2Vec topics. Related scripts are `extractTweetGroupsWithMostImportantWords.py`, `findMostImportantTopicTweets.py`, `getScoresAndTopicsForAManualTopic.py` and `predictSingleTweetGroup.py`.

### 8. Topic model evaluation and LLM comparison

```bash
python evaluateTopicModel.py
```

The script reports the number of topics, the topic diversity and the coherence. The ChatGPT comparison presented in Publication 1 is reproduced as follows:

- `createTweetGroupDataframe.py` exports the test tweet groups as CSV.
- `selectTweetGroupsFromTweetGroupDataframe.py` samples a balanced and shuffled subset for prompting the LLM.
- `comparePredictionLSTMAndLLMPrediction.py` scores the LLM predictions, read back from a text file, with the same metrics as the LSTM.
- `compareTopicModelsAndLLMTopics.py` and `extractFeatureTagsFromLLMAndLSTM.py` compare the topics and feature tags of the topic models against those of the LLM.

### Running the tests

```bash
python -m unittest tests.alltestsuite
```

---

## Notes

- Paths in the code are Windows-style and absolute to the author's data directory. `DataDirHelper` and the few scripts with hard-coded paths have to be adapted to the local environment.
- The datasets, trained checkpoints and topic models are **not** part of this repository, in line with Twitter's terms on redistributing content. They have to be rebuilt with the steps described above.
- Results depend on the tweet group size and the class mapper. Both are defined by the `PredictionModelPath` constant that is selected.
