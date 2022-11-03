import hydra
import json
import logging
import os
import pandas as pd
import torch

from glob import glob
from omegaconf import DictConfig
from ptls.frames.supervised.metrics import BucketAccuracy, RankAUC
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError, R2Score

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(conf: DictConfig):
    # hydra.initialize(version_base=None, config_path="conf")
    # conf = hydra.compose("pl_regressor", overrides=["device=cpu"])
    cols = [conf.data_module.setup.col_id, conf.monte_carlo.benchmark.col, conf.monte_carlo.col]
    bm = pd.concat([pd.read_csv(fn) for fn in glob(os.path.join(conf.monte_carlo.benchmark.dir, "version_*/prediction.csv"))], axis=0)
    bm.rename(columns={"seq_id": cols[0], "out": "benchmark"}, inplace=True)
    tab = pd.read_csv(os.path.join(conf.work_dir, "monte_carlo.csv"))[cols].merge(bm, on=cols[0])
    logger.info(f"Benchmark {bm.shape} and joint table {tab.shape} collected.")

    tg = torch.tensor(tab[cols[1]].values, dtype=torch.float32)
    mc = torch.tensor(tab[cols[2]].values, dtype=torch.float32)
    bm = torch.tensor(tab["benchmark"].values, dtype=torch.float32).exp() - 1

    res = {m.__class__.__name__: {"benchmark": m(bm, tg).item(), "ilmc": m(mc, tg).item()}
           for m in (BucketAccuracy(), MeanAbsoluteError(), MeanAbsolutePercentageError(),
                     RankAUC(), R2Score(), SymmetricMeanAbsolutePercentageError())}
    logger.info("\n" + json.dumps(res, indent=2))
    with open(os.path.join(conf.work_dir, "monte_carlo.json"), "w") as fo:
        json.dump(res, fo, indent=2)


if __name__ == "__main__":
    main()
