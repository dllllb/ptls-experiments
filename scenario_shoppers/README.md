# Get data

```sh
cd scenario_shoppers

# download dataset
sh bin/get-data.sh

# convert dataset from transaction list to features
sh bin/make-dataset.sh
```

# Main scenario, best params

```sh
export CUDA_VISIBLE_DEVICES=0

sh bin/scan_loss.sh
sh bin/scan_hparams.sh

# one must insert actual paths and parameters into the script
sh bin/apply.sh
```
