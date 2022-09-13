# About the task
[Predict default of client using his credit history data](https://ods.ai/competitions/dl-fintech-bki) (binary classification).

All features are categorical, numerical features already discretized. The dataset is arranged in such a way that credits for the training sample are taken over a period of M months, and credits for the test sample are taken over the next K months.

# About the data
1. {train/test}_target.csv – training and test sample of loans:
- id - identifier of the loan application
- flag - target variable, 1 - the fact that the client went into default (available only in the train_target.csv file)

2. {train/test}_data - credit history data for training and training the model:
- id - request identifier
- rn is the serial number of the loan product in the credit history
- pre_since_opened - days from the loan opening date to the data collection date
- pre_since_confirmed - days from the date of confirmation of information on the loan to the date of data collection
- pre_pterm - the planned number of days from the opening date of the loan to the closing date
- pre_fterm - the actual number of days from the opening date of the loan to the closing date
- pre_till_pclose - the planned number of days from the data collection date to the loan closing date
- pre_till_fclose - the actual number of days from the data collection date to the loan close date
- pre_loans_credit_limit - credit limit
- pre_loans_next_pay_summ - the amount of the next loan payment
- pre_loans_outstanding - remaining unpaid loan amount
- pre_loans_total_overdue - current overdue debt
- pre_loans_max_overdue_sum - maximum overdue debt
- pre_loans_credit_cost_rate - total cost of the loan
- pre_loans5 - the number of delays up to 5 days
- pre_loans530 - the number of delays from 5 to 30 days
- pre_loans3060 - the number of delays from 30 to 60 days
- pre_loans6090 - the number of delays from 60 to 90 days
- pre_loans90 - the number of overdue loans for more than 90 days
- is_zero_loans_5 - flag: no delays up to 5 days
- is_zero_loans_530 - flag: no delays from 5 to 30 days
- is_zero_loans_3060 - flag: no delays from 30 to 60 days
- is_zero_loans_6090 - flag: no delays from 60 to 90 days
- is_zero_loans90 - flag: no delays for more than 90 days
- pre_util - the ratio of the remaining unpaid loan amount to the credit limit
- pre_over2limit - the ratio of the current overdue debt to the credit limit
- pre_maxover2limit - the ratio of the maximum overdue debt to the credit limit
- is_zero_util - flag: the ratio of the remaining unpaid loan amount to the credit limit is 0
- is_zero_over2limit - flag: the ratio of the current overdue debt to the credit limit is 0
- is_zero_maxover2limit - flag: the ratio of the maximum overdue debt to the credit limit is 0
- enc_paym_{0..n} – monthly payment statuses for the last n months
- enc_loans_account_holder_type - type of loan relationship
- enc_loans_credit_status - loan status
- enc_loans_account_cur - loan currency
- enc_loans_credit_type - type of loan
- pclose_flag - flag: the planned number of days from the opening date of the loan to the closing date is not defined
- fclose_flag - flag: the actual number of days from the opening date of the loan to the closing date is not defined

# Get data

```sh
cd scenario_alpha_rnn_vs_transformer

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
python make_dataset.py
```

# Run models:

```
export CUDA_VISIBLE_DEVICES=0

# rnn
python rnn_model.py

# transformer
python transf_model.py

# check the results
tensorboard --logdir=lightning_logs/ --port=6006
```
