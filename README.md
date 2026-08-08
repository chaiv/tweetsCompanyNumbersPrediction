# Predicting Company Financial Metrics from Tweets

Research code for the PhD thesis *"AI-powered prediction of organizational financial metrics from social media data"* by Vitali Chaiko.

## Abstract

This research aims to improve organizational management and processes by designing an AI-powered automated system for the prediction of various organizational financial metrics based on machine learning and social media data. In addition to predictive functionality, the system is designed to extract the most important social and economic topics and keywords that contribute to the prediction performance. The concept of the research follows an exploratory step-by-step approach, resulting in a quantitative evaluation.

The research subject of this thesis consists of multiple datasets:

1. Publicly accessible **Twitter data** for five leading NASDAQ-listed U.S. companies — Amazon, Apple, Google, Microsoft and Tesla — from 2015 to 2020.
2. Public **financial metrics** associated with those companies, such as quarterly revenue.
3. Socially relevant **topics and social issue keywords** visible to the public from 2015 to 2020.

Custom neural networks and topic models were trained successfully, demonstrating the practicability and generalization of predictions on financial metrics based on text information, and achieving high evaluation metrics, including an **accuracy of 0.87** and a **Matthews correlation coefficient of 0.77**. Meaningful insights — such as a connection between the Black Lives Matter movement and Amazon's Christmas campaign, or a Brexit-driven investor shift to cryptocurrency — can be extracted by the system.

Directions for future research include improving the performance of topic models, which underperformed compared to prediction models (UCI coherence score of 0.44), and real-world testing of the system in context-specific use cases with domain experts.

## Publications & Resources

| Resource | Link |
| --- | --- |
| PhD thesis (PDF) | [EN – Dissertation – Vitali Chaiko](https://ras.nacid.bg/api/reg/FilesStorage?key=6db2fa58-4805-4803-9d11-3dc09e838e6e&mimeType=application/pdf&fileName=EN%20-%20Dissertation%20-%20Vitali%20Chaiko%20-%2010.03.2026.pdf&dbId=2) |
| Thesis bibliographic record | [COBISS 79716360](https://plus.cobiss.net/cobiss/bg/en/data/cobib/79716360) |
| Publication 1 | [Comparing ChatGPT And LSTM In Predicting Changes In Quarterly Financial Metrics](https://www.researchgate.net/publication/381910297_Comparing_ChatGPT_And_LSTM_In_Predicting_Changes_In_Quarterly_Financial_Metrics) |
| Publication 2 | [Buditeli proceedings 2024 (UNIBIT)](https://buditeli.unibit.bg/images/proceedings/2024/chaiko.pdf) |
| Publication 3 | [UNIBIT e-journal proceedings 2024, book 2](https://e-journal.unibit.bg/images/proceedings/2024/book2/5_Statiq_Vitali_Chaiko.pdf) |
| Primary dataset | [Tweets about the Top Companies from 2015 to 2020 (Kaggle)](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) |

---

## How the System Works

The pipeline turns raw tweets into a supervised classification problem:

1. **Tweets are joined with financial figures.** Every tweet is labelled with the change of the company's next reported financial metric (quarterly revenue, EPS, car sales, search-engine market share), so the label describes what happened *after* the tweet was posted.
2. **The percentage change is discretized into classes** — either binary (`decrease` / `increase`) or 4-class (`decrease`, `weak`, `moderate`, `strong increase`).
3. **Tweets are bundled into "tweet groups"** of N tweets sharing the same class (N = 5, 10 or 20). A group is one training sample; single tweets carry too little signal.
4. **An LSTM classifier** over Top2Vec-derived word embeddings predicts the class of a tweet group.
5. **Integrated Gradients (Captum)** attributes the prediction back to individual tokens, and **Top2Vec / BERTopic** models map those tokens to human-readable topics.

## Repository Structure

Everything lives under [`tweetsCompanyNumbersPrediction/src`](tweetsCompanyNumbersPrediction/src). Top-level `.py` files are *runnable scripts*; the sub-packages contain the reusable library code.

### Configuration

| File | Purpose |
| --- | --- |
| [`PredictionModelPath.py`](tweetsCompanyNumbersPrediction/src/PredictionModelPath.py) | Central experiment registry. Each constant (e.g. `AMAZON_REVENUE_10_LSTM_MULTI_CLASS`) bundles the NASDAQ tag, dataset paths, word-vector path, model checkpoint directory, topic-model path, tweet group size and class mapper for one experiment. **Every script selects its experiment by assigning one of these constants to `predictionModelPath`.** |
| [`TrainingMode.py`](tweetsCompanyNumbersPrediction/src/TrainingMode.py) | Enum of the four evaluation strategies (`SUBSEQUENT`, `STRATIFIED_TEMPORAL`, `TEMPORAL_SPLIT`, `STRATIFIED_KFOLD_TEMPORAL_PER_CLASS`) and whether they sort tweets temporally. |
| [`tweetpreprocess/DataDirHelper.py`](tweetsCompanyNumbersPrediction/src/tweetpreprocess/DataDirHelper.py) | Resolves the data root directory. **Adjust the paths here before running anything.** |

### Library packages

| Package | Contents |
| --- | --- |
| `tweetpreprocess` | Tweet querying/sorting, date→timestamp conversion, text and stop-word filtering (`wordfiltering/`), near-duplicate detection, class calculators (`FiguresIncreaseDecreaseClassCalculator`, `FiguresMultiClassCalculator`, `FiguresPercentChangeCalculator`) and `EqualClassSampler` for class balancing. |
| `tweetnumbersconnector` | Joins each tweet with the financial figure of its reporting period (`TweetNumbersConnector`) and derives the target classes (`FinancialFiguresClassifier`). |
| `pipeline` | `FeatureDataframePipeline` — the end-to-end composition of the preprocessing steps into the labelled tweet dataframe. |
| `nlpvectors` | Tokenization (`TweetTokenizer`), vocabulary creation, word-vector/ID encoders, `DataframeSplitter` (builds the tweet groups), `TweetGroup` and dataframe conversion helpers. |
| `classifier` | `LSTMNN` / `LSTMWithDropout` (PyTorch Lightning modules), `CreateClassifierModel` (model factory), `Trainer` (Lightning training loop with early stopping, checkpointing, TensorBoard), `TweetGroupDataset`, `ClassificationMetrics`, `PredictionClassMappers` (`BINARY_0_1`, `MULTICLASS_4`) and `transformer/Predictor` for batched inference. |
| `featureinterpretation` | `AttributionsCalculator` (Captum Integrated Gradients), `ImportantWordsStore`, `TokenScoresSort` and dataframe utilities for the most-important-word analysis. |
| `topicmodelling` | `TopicModelCreator` / `TopicExtractor` for Top2Vec and BERTopic, `TopicEvaluation` (coherence, diversity), `ManualTopicAnalyzer` and `llmcomparison/LLMTopicsCompare`. |
| `exploredata` | Descriptive statistics and plots for the tweet and financial dataframes, plus POS tagging. |
| `sentiment` | VADER-based sentiment analysis of tweets and words. |
| `tests` | Unit tests; [`tests/alltestsuite.py`](tweetsCompanyNumbersPrediction/src/tests/alltestsuite.py) is the aggregated suite. |

---

## Reproducing the Experiments

### 1. Setup

```bash
git clone https://github.com/chaiv/tweetsCompanyNumbersPrediction.git
```

```bash
pip install -r tweetsCompanyNumbersPrediction/requirements.txt
```

Python 3.10 with a CUDA-capable GPU is assumed (`Trainer` defaults to `accelerator='cuda'`; change it to `'cpu'` if needed). Scripts are run from the `src` directory, which must be on `PYTHONPATH`:

```bash
cd tweetsCompanyNumbersPrediction/src
```

### 2. Prepare the data directory

Download the [Kaggle dataset](https://www.kaggle.com/datasets/omermetinn/tweets-about-the-top-companies-from-2015-to-2020) and merge `Tweet.csv` with `Company_Tweet.csv` into `CompanyTweets.csv` using [`tweetpreprocess/companyTweetMerge.py`](tweetsCompanyNumbersPrediction/src/tweetpreprocess/companyTweetMerge.py).

Place it — together with a CSV of the financial metric per company (columns include the reporting date and `percent_change`) — in a `companyTweets` folder inside your data root, then point `DataDirHelper` at that root. The expected file names per experiment are listed in `PredictionModelPath.py`.

### 3. Build the labelled tweet dataframe

```bash
python createTweetsWithNumbers.py
```

Joins tweets with the financial figures and writes the labelled dataframe to `predictionModelPath.getDataframePath()`. Select the experiment at the top of the script.

### 4. Train the topic model and export word vectors

```bash
python trainTop2VecTopicModel.py
```

```bash
python top2VecWordVectorsToFile.py
```

The first trains a Top2Vec model on the tweet corpus; the second exports its vocabulary as word2vec-format vectors (plus `<PAD>`, `<UNK>`, `<SEP>` tokens) to `getWordVectorsPath()` — these are the pretrained embeddings of the LSTM. Optionally run `createTokenizerLookup.py` to build the token→original-word lookup used by the interpretation scripts, and `trainBertTopicModel.py` for the BERTopic alternative.

### 5. Train the prediction model

Pick the evaluation strategy that matches the research question:

| Script | Split strategy |
| --- | --- |
| `trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py` | 10-fold stratified CV with rotating temporal test blocks per class — the main evaluation. |
| `trainNumbersPredictionModelStratifiedExpandingWindowPerClass.py` | Expanding window per class: train always temporally before test (no future leakage). |
| `trainNumbersPredictionModelTemporalSplit.py` | Strict 80/20 temporal split — closest to a real forecasting setting. |
| `trainNumbersPredictionModelStratifiedTemporalOrder.py` | Latest tweets as test set, stratified. |
| `trainNumbersPredictionModelOnlySubsequentTweetsOrder.py` | Groups of N subsequent tweets without a temporal test set — used for the topic and important-word analysis. |

```bash
python trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py
```

Each fold writes a checkpoint `tweetpredict_fold{k}.ckpt` and its test indices `test_idx_fold{k}.npy` into `getModelPath()`, prints a per-fold classification report and MCC, and finishes with the mean MCC across folds. TensorBoard logs go to `companyTweets/modellogs`.

### 6. Evaluate

```bash
python calculateClassificationMetrics.py
```

Reloads a checkpoint and its saved test split, and reports precision/recall/F1, accuracy and MCC. `TRAINING_MODE` must match the training script that produced the checkpoint, since it controls whether tweet groups are sorted temporally.

### 7. Extract the most important words and topics

```bash
python extractMostImportantWords.py
```

```bash
python calculateTopImportantWordsWithTopics.py
```

The first computes Integrated Gradients attributions per token for a chosen observed class and stores them as `importantWordsClass{N}{Company}.csv`. The second ranks them, adds the untokenized word, POS tags and the matching Top2Vec topics. Related: `extractTweetGroupsWithMostImportantWords.py`, `findMostImportantTopicTweets.py`, `getScoresAndTopicsForAManualTopic.py`, `predictSingleTweetGroup.py`.

### 8. Topic model evaluation and LLM comparison

```bash
python evaluateTopicModel.py
```

Prints the number of topics, topic diversity and coherence. For the ChatGPT comparison published in Publication 1:

- `createTweetGroupDataframe.py` → exports the test tweet groups as CSV.
- `selectTweetGroupsFromTweetGroupDataframe.py` → samples a balanced, shuffled subset to prompt the LLM with.
- `comparePredictionLSTMAndLLMPrediction.py` → scores the LLM predictions (read back from a text file) with the same metrics as the LSTM.
- `compareTopicModelsAndLLMTopics.py` and `extractFeatureTagsFromLLMAndLSTM.py` → compare topics and feature tags of the topic models against the LLM.

### Running the tests

```bash
python -m unittest tests.alltestsuite
```

---

## Notes

- Paths throughout the code are Windows-style and absolute to the author's data directory; adapt `DataDirHelper` (and the few scripts with hard-coded paths) to your environment.
- The datasets, trained checkpoints and topic models are **not** part of this repository in accordance to Twitter content distribution  — they must be rebuilt with the steps above.
- Results depend on the tweet group size and the class mapper; both are part of the `PredictionModelPath` constant you select.
