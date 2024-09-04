Experiments on public datasets for `pytorch-lifestream` library

# Setup and test using pipenv

```sh
# Ubuntu 18.04

sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv sync  --dev # install packages exactly as specified in Pipfile.lock
pipenv shell
pytest

# run luigi server
luigid
# check embedding validation progress at `http://localhost:8082/`

# use tensorboard for metrics exploration
tensorboard --logdir lightning_logs/ 
# check tensorboard metrics at `http://localhost:6006/`

```

# Run scenario

 We check 5 datasets as separate experiments. See `README.md` files in experiments folder:
 
 - [Age](scenario_age_pred/README.md)
 - [Churn](scenario_rosbank/README.md)
 - [Assess](scenario_bowl2019/README.md)
 - [Retail](scenario_x5/README.md)
 - [Scoring](scenario_alpha_battle/README.md)
 - [Small demo dataset](scenario_gender/README.md)

See also [additional list of experiments](#other-experiments)

# Notebooks

Full scenarious are console scripts configured by hydra yaml configs.
If you like jupyter notebooks you can see an example for AgePred dataset in [AgePred notebooks](scenario_age_pred/notebooks/)

# Results

All results are stored in `*/results` folder.

[See detailed results tables](results.md)

# Other experiments

- [Data Fusion Contest 2024, 2-st place on the Churn Task](https://github.com/warofgam/Sber-AI-Lab---datafusion) (in Russian) 
- [Data Fusion Contest 2024, Ivan Alexandrov](https://github.com/Ivanich-spb/datafusion_2024_churn) (in Russian)
- [Data Fusion Contest 2022. 1-st place on the Matching Task](https://github.com/ivkireev86/datafusion-contest-2022)
- [Alpha BKI dataset](experiments/scenario_alpha_rnn_vs_transformer/README.md)
- [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction)
    - [Supervised training with RNN](https://www.kaggle.com/code/ivkireev/amex-ptls-baseline-supervised-neural-network)
    - [Supervised training with Transformer](https://www.kaggle.com/code/ivkireev/amex-transformer-network-train-with-ptls)
    - [CoLES Embedding preparation](https://www.kaggle.com/code/ivkireev/amex-contrastive-embeddings-with-ptls-coles)
    - [CoLES Embedding usage as extra features for catboost](https://www.kaggle.com/code/ivkireev/catboost-classifier-with-coles-embeddings)
- [Softmax loss](experiments/softmax_loss_vs_contrastive_loss/readme.md) - try CoLES with Softmax loss.
- [Random features](experiments/random_features/readme.md) - how CoLES works with slowly changing features which helps to distinguish clients.
- [Small prretrain](experiments/mles_experiments_supervised_only/README.md) - check the CoLES quality depends on prertain size.
- [ILMC for aggregate values estimation](scenario_shoppers/README.md) - Imitation Learning Monte-Carlo for CLTV on Acquire Valued Shoppers Challenge dataset
- [COTIC](https://github.com/VladislavZh/COTIC) - `pytorch-lifestream` is used in experiment for Continuous-time convolutions model of event sequences.

