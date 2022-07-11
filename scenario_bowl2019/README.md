# Get data

```sh
cd experiments/scenario_bowl2019

# download datasets
sh bin/get-data.sh

# convert datasets from transaction list to features for metric learning
sh bin/make-datasets-spark.sh
```

# Main scenario, best params

```sh
cd experiments/scenario_bowl2019
export CUDA_VISIBLE_DEVICES=0

sh bin/run_all_scenarios.sh

# check the results
cat results/*.txt
cat results/*.csv
```