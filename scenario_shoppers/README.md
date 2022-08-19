# Data preprocessing

The dataset represents purchasers' activity timeline. Each transaction may be encoded by
a pair of one categorical (generalized shop/brand/product code) and one numerical (priced
quantity) variable. Further we combine a number of these transactions within a certain
time span into one multinomial variable (denoted here as aggregated purchase vector, APV).

The variety of such transactional data can be effectively handled via a combination of a
trx-encoder (which provides concatenated embeddings for pairs of cat/num variables) and a
seq-encoder (provides RNN-encoded embeddings of the whole event sequence).

```shell
# https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
# download and preprocess dataset (and a sample from it) into event sequences with
# properly defined state and corresponding target variables
# useful statistics will be saved in <make_dataset.log>

cd ./scenario_shoppers/
sh bin/make_dataset.sh
```

# Benchmark and one-step predictors

The main goal of the task is to represent a long range target variable (e.g. CLTV) via
modeling APV-distribution over shorter time span. In practice one may consider 120-day
CLTV and its evaluation by 12 one-step predictions of corresponding 10-day APV.

```shell
# scanning over loss-functions and hyperparameters
# compute faster overriding conf variable {data_path} by "data_path=data/sample"

sh bin/scan_loss.sh
sh bin/scan_hparams.sh

# after those one must analyze <cv_result.json> and insert the actual checkpoint
# numbers {*_CKPT} into <bin/apply.sh> for every benchmark and one-step predictors
```

# Imitation Learning Monte-Carlo

As long as one-step predictor is learned one can evaluate CLTV as the expectation of sum
of APV over continuum of possible 12-step trajectories. We compute the expectation by
Monte-Carlo method.

```shell
# ILMC requires a sufficient number of {repeats} and proper {chunk} size
# ILMC results saved in <monte_carlo.csv>; metrics (MAE, BucketAccuracy, RankAUC,
# Jensen-Shannon divergence, MAPE, SMAPE, R2) saved in <monte_carlo.json>

sh bin/apply.sh
```
