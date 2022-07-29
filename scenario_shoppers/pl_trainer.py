import json
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


def fold_fit_test(conf, fold_id):
    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model, fold_id=fold_id)

    trainer = pl.Trainer(callbacks=[TQDMProgressBar(10)], **conf.trainer,
                         logger=TensorBoardLogger(save_dir=conf.get("logger_name"), name=None))
    trainer.fit(model, datamodule=dm)
    valid_metrics = {name: float(mf.compute().item()) for name, mf in model.valid_metrics.items()}

    trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path="best", verbose=False)
    test_metrics = {name: float(mf.compute().item()) for name, mf in model.test_metrics.items()}

    print(", ".join([f"valid_{name}: {v:.4f}" for name, v in valid_metrics.items()]))
    print(", ".join([f" test_{name}: {v:.4f}" for name, v in test_metrics.items()]))

    res = trainer.predict(model=model, dataloaders=dm.predict_dataloader(), ckpt_path="best")
    pd.concat(res, axis=0).to_csv(f"{trainer.log_dir}/test_prediction.csv", header=True, index=False)

    return {"fold_id": fold_id,
            "model_name": conf.embedding_validation_results.model_name,
            "feature_name": conf.embedding_validation_results.feature_name,
            "scores_valid": valid_metrics,
            "scores_test": test_metrics}


@hydra.main(version_base=None)
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)
    pl.seed_everything(conf.seed_everything)

    with open(conf.data_module.setup.fold_info) as fo:
        fold_info = json.load(fo)
    fold_list = [k for k in sorted(fold_info.keys()) if not k.startswith('_')]

    res = [fold_fit_test(conf, fold_id) for fold_id in fold_list]
    with open(conf.embedding_validation_results.output_path, "w") as fo:
        json.dump(res, fo, indent=2)

    for name in res[0]["scores_valid"].keys():
        valid_scores = np.array([x["scores_valid"][name]for x in res])
        test_scores = np.array([x["scores_test"][name] for x in res])
        print(f'Valid {name:10}: {valid_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in valid_scores)}], '
              f'Test  {name:10}: {test_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in test_scores)}]')


if __name__ == "__main__":
    main()
