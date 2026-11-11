# Party in Thee Code

Kaggle team: **Party in Thee Code**.

Current contributor: Skye Li (yl4534), baseline and Q1/Q2. The other team
members and their contributions will be added with their work.

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

## AI assistance

ChatGPT helped draft and debug the baseline and Q1/Q2 code and notes,
execute the experiments and check the exported outputs. The final report will also describe the team's
own checks and later changes.

## Current status

Q1 comparison and tuning are complete. The current Q1 choice is character
TF-IDF + linear SVM (C=1), with training-only sigmoid calibration for probability
output. Its validation AUROC is 0.970704. Q1 results are under results/q1/.
Q2 learning curves are also complete; see Q2_README.md and results/q2/.
Teammate experiments and the final Kaggle prediction command are still in progress.

## Q2 learning curves

Use `python -m pip install -r requirements_q2.txt`, then
`python q2.py --train data/train.csv`, or open q2_learning_curve.ipynb.
The original split and Q1 settings are reused. Results and both figure formats
are saved under results/q2/.
