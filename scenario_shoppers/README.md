# Data preprocessing

The dataset represents purchaser's activity timeline; each transaction may be
encoded by one pair of a categorical (a combined shop/brand/product code) and a numerical
(a priced quantity) variable. Herein we denote a sum of this paired variables over a
certain time span as aggregated purchase vector (APV). The variety of such transactional
data can be effectively handled via a combination of a trx-encoder (which provides
concatenated embeddings for pairs of cat/num variables) and a seq-encoder (provides
RNN-encoded embeddings of the whole event sequence).

```shell
# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
# download and preprocess dataset (and a sample from it) into event sequences with
# properly defined state and corresponding target variables
# useful statistics will be saved in <make_dataset.log>

cd ./scenario_shoppers/
sh bin/make_dataset.sh
```

# Benchmark and one-step predictors

The main goal of the task is to represent a long range target variable (e.g. CLTV)
via modeling APV-distribution over shorter time span. For instance, in practice one may
consider 120-day CLTV and its evaluation by 12 one-step predictions of corresponding
10-day APV.

```shell
# scanning over loss-functions and hyperparameters
# compute faster overriding conf variable {data_path} by "data_path=data/sample"

sh bin/scan_loss.sh
sh bin/scan_hparams.sh

# after those one must insert actual {WORK_DIR} and {CKPT} into <bin/apply.sh>
# for both benchmark and one-step predictors
```

# Imitation Learning Monte-Carlo

If one-step predictor is learned one can evaluate CLTV as expectation of sum of APV over
continuum of possible 12-step trajectories. We compute the expectation by Monte-Carlo
method.

```shell
# ILMC requires a sufficient number of {repeats} and proper {chunk} size
# ILMC results saved in <monte_carlo.csv>; metrics (MAE, BucketAccuracy, RankAUC,
# Jensen-Shannon divergence, MAPE, SMAPE, R2) saved in <monte_carlo.json>

sh bin/apply.sh
```
