# Get data

```sh
cd scenario_age_pred

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make_datasets_spark_file.sh
```

# Main scenario, best params

```sh
cd scenario_age_pred
export CUDA_VISIBLE_DEVICES=0

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```
