# Q1: model comparison

Party in Thee Code. Q1.1-Q1.4 results are in the separate Overleaf snippet.
The original baseline code, split files and baseline predictions are unchanged.
Q2 and the later team experiments are separate work.

## Reproduce

Use the same dependencies as the baseline (`requirements.txt`, Python 3.11+).
Put the supplied train.csv in data/; train(1).csv can be renamed to train.csv.
From this team folder:

```bash
python -m pip install -r requirements.txt
python q1.py --train data/train.csv
```

Alternatively, open q1_model_comparison.ipynb beside baseline.py and execute
its cells. It calls the same script functions. The included notebook has
actual executed outputs already. Do not run both entrypoints unless a second
run is wanted. Fit time varies by machine. Running Q1 regenerates results/q1/
and models/q1/, while leaving the original baseline results and split IDs alone.
All fits limit BLAS to one thread.

## What was tried

- Word TF-IDF + Logistic Regression: C = 0.1, 1, 10.
- Word TF-IDF + Multinomial Naive Bayes: alpha = 0.1, 0.5, 1.
- Character TF-IDF + Linear SVM: C = 0.1, 1, 10.

Word TF-IDF uses the baseline's unigrams/bigrams and settings. Character TF-IDF
uses character 3-5-grams. Both use min_df=2, max_features=50000, sublinear TF,
smoothed IDF and L2 normalization. Each candidate learns its own features from
the 8,000 training texts. No unlabeled or Kaggle test texts are read by Q1.
Each family is tuned using the same shared validation set, so the best score
is a selection score rather than an unbiased final estimate.

Current selected pipeline: character SVM, C=1. Validation AUROC:
0.970704. For probabilities, it uses training-only 3-fold
sigmoid calibration with ensemble=False. The cross-validation wraps the full
TF-IDF pipeline, so each fold learns features only from its training portion.
The final base estimator is fitted on all 8,000 training rows. Model-comparison
times exclude this separate calibration fit (65.878 s in this run).

## Files

- q1.py: experiment code and reusable model constructors.
- q1_model_comparison.ipynb: executed comparison notebook.
- results/q1/hyperparameter_results.csv: all nine configurations.
- results/q1/model_comparison.csv: best configuration from each family.
- results/q1/selection.json: selected setting, calibration and split hashes.
- results/q1/model_configs.json: complete estimator settings.
- results/q1/*_validation_scores.csv: each family's selected validation scores.
  char_svm uses decision_score, which is not a probability.
- results/q1/selected_validation_predictions.csv: calibrated P(AI), labels, IDs,
  word counts and error flags for the selected Q1 model.
- results/q1/selected_training_predictions.csv: matching training predictions.
- models/q1/selected_model.joblib: generated locally, excluded from GitHub files.

The new validation file is available for Q4 if the team adopts this Q1 model.
Keep results/validation_predictions.csv as the original baseline output.
For Q3/Q5, compare changes against whichever supervised configuration is
explicitly chosen for that experiment; do not silently compare different baselines.

To fit just the selected probability model without rerunning the nine candidates:

```python
from baseline import load_data, load_split
from q1 import make_probability_model

train, val = load_split(load_data("data/train.csv"))
model = make_probability_model("char_svm", 1.0)
model.fit(train.text, train.label)
p_ai = model.predict_proba(val.text)[:, 1]
```

The model is selected for Q1, not yet retrained on all 10,000 rows for Kaggle.
The final team choice can change after Q3/Q5.

AI assistance: ChatGPT helped write and debug Q1 code, execute the experiments,
check outputs and format the LaTeX. The report should reflect the team's later
review and changes as well.

Implementation references:
- https://scikit-learn.org/1.8/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- https://scikit-learn.org/1.8/modules/compose.html
