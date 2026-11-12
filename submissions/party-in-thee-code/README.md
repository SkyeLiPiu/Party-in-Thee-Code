# Party in Thee Code

Kaggle team: **Party in Thee Code**.

Team members:

- Skye Li (yl4534): shared split, baseline, Q1 model comparison, Q2 learning curves.
- Candice (GitHub Candice0313, yx772): Q3 pseudo-labeling, and the character 2-5 gram plus blank-line submission.
- James Chen (jzc28): Q4 error analysis, Q5 improvement, and the newline pipeline in predict.py.

Team branch: `submission/party-in-thee-code`. Project folder:
`submissions/party-in-thee-code/`. Run the commands below from this folder.

The original baseline remains available below. Q1 model comparison and tuning
are now included in q1.py and q1_model_comparison.ipynb; see Q1_README.md.

Python 3.11 or newer is needed for the pinned packages. The exact run environment
is in `results/environment.json`.

```bash
python -m pip install -r requirements.txt
```

Put the supplied `train.csv` in `data/`. The downloaded name `train(1).csv`
can just be renamed. Open `baseline.ipynb` from this folder in Jupyter, or use:

```bash
python baseline.py --train data/train.csv
```

The notebook has executed outputs already. Rerunning saves a new fit time and
predictions. Timing depends on the machine, so compare runtimes from the same
environment for later experiments.

The split is 8,000 training rows and 2,000 validation rows, stratified with seed
42. Training has 4,000 examples per label; validation has 1,000 per label.
`splits/train_ids.csv` and `splits/validation_ids.csv` are the team split.
Load these files for later experiments rather than making another split.
The IDs were split after sorting them as strings; rerunning train_test_split on
the original CSV order would give a different split. The manifest also checks
the labeled data fingerprint, ignoring filename and row order.

The baseline uses lowercase word TF-IDF with unigrams and bigrams, min_df=2,
up to 50,000 features, sublinear term frequency and L2 normalization.
Tokenization keeps Unicode word tokens of at least two characters. There is
no stemming, stopword removal, accent stripping or separate punctuation feature.
Logistic regression uses C=1, liblinear, max_iter=2000, tol=1e-4 and seed 42.
This configuration was fixed before evaluation; no parameter search was done.
Only the 8,000 training texts fit the vocabulary, IDF and classifier.

Useful files:

- `baseline.py`: shared data, split, model and evaluation functions.
- `baseline.ipynb`: executed baseline, split checks and metric output.
- `results/baseline_metrics.csv`: training AUROC, validation AUROC and fit time.
- `results/validation_predictions.csv`: IDs, labels, AI probabilities, word
  counts and error types. No full passages are copied into this file.
- `results/training_predictions.csv`: matching training predictions.
- `results/data_audit.json`: missing-value and duplicate checks.
- `baseline_config.json`: exact feature and classifier settings.
- `models/baseline.joblib`: generated locally by training. The model checkpoint
  is available in the private handoff ZIP and is not committed here.
- `HANDOFF.md`: examples for Q3-Q6.

The saved model was fitted on 8,000 rows. It does not use unlabeled.csv or test.csv
for fitting. No Kaggle submission or test score is included. Exact and
case/whitespace-normalized duplicates were checked; broader near-duplicate and
source leakage were not assessed because source metadata are unavailable.

Raw CSVs and model checkpoints are excluded from this repository and ignored by git. Everyone
can use their own copy of the supplied data with the same shared ID lists.

## Q3 to Q5 and the final submission

Put train.csv, unlabeled.csv and test.csv in data/. Each command runs from this
folder with `requirements_q2.txt` installed, uses seed 42 and the shared split,
and limits BLAS to one thread. Results go to results/ as JSON or Markdown.

```bash
python q3_pseudo_label.py --train data/train.csv   # Q3, writes results/q3_pseudo_metrics.*
python q4_error_analysis.py --train data/train.csv # Q4, writes results/q4/ (IDs only, no passages)
python q5_improvement.py --train data/train.csv    # Q5 round 1 (v1-v3), about 3 min
python q5_round2.py --train data/train.csv         # Q5 round 2 (v4-v6), about 1 min
python predict.py --train data/train.csv --validate   # Q5 newline model, 8,000-row check
python q5_char_blank.py --train data/train.csv        # current model, shared-split comparison
python q5_char_blank.py --train data/train.csv --test data/test.csv --out submission.csv
```

`q5_char_blank.py --test` is the current submission command. It fits on all
10,000 labeled rows and writes `submission.csv`. The model is a character
2-5 gram SVM plus a blank-line indicator. Combination weights come from
3-fold out-of-fold scores on the training rows. On the shared validation
split the clean AUROC is 0.9937, against 0.9917 for the newline model.
With whitespace collapsed the scores are 0.9261 and 0.8887. The public
leaderboard score for this submission is 0.95836.

`predict.py` is the earlier Q5 pipeline. Its test run writes the newline-aware
character SVM and two comparison files. That is not the current submission.

Formatting artifact: in train.csv, 0% of human passages contain a blank line,
against 66% of AI passages. scikit-learn's character analyzer collapses blank
lines, so the Q1 model never saw this. The newline model turns each newline
into "¶" and reaches 0.9917, but collapsed whitespace drops it to 0.889.
The current model keeps that cue as one feature. 31.9% of test.csv passages
contain a blank line (33.0% of train.csv). Collapsed whitespace still lowers
the current model, to 0.9261, and the public score is below the clean
validation number.

## AI assistance

ChatGPT helped draft and debug the baseline and Q1/Q2 code and notes,
execute the experiments and check the exported outputs. Claude (Anthropic)
helped write and run the Q4, Q5 and predict.py code and draft the
corresponding report text. The team reviewed the code, reran the commands and
checked the numbers against the saved outputs.
No AI tool or external API labeled or scored the competition passages.

## Current status

- Q1: character TF-IDF + linear SVM (C=1), validation AUROC 0.970704 (results/q1/).
- Q2: learning curves (Q2_README.md, results/q2/).
- Q3: pseudo-labeling with the word LR baseline, no gain (results/q3_pseudo_metrics.json).
- Q4: error analysis of the Q1 model (results/q4/).
- Q5: newline-aware character SVM, validation AUROC 0.991701. Collapsed whitespace falls to 0.889 (results/q5/). `predict.py` still runs this model.
- Current submission: character 2-5 grams plus a blank-line feature (`q5_char_blank.py`). Shared-split AUROC 0.9937 clean and 0.9261 with whitespace collapsed. Public leaderboard 0.95836.

## Q2 learning curves

Use `python -m pip install -r requirements_q2.txt`, then
`python q2.py --train data/train.csv`, or open q2_learning_curve.ipynb.
The original split and Q1 settings are reused. Results and both figure formats
are saved under results/q2/.
