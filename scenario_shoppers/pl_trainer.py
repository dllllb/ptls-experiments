import json
import logging
import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)


def fold_fit_test(conf, fold_id):
    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model, fold_id=fold_id)

    extra_params = {}
    if "logger_name" in conf:
        extra_params["logger"] = TensorBoardLogger(save_dir="lightning_logs", name=conf.get("logger_name"))
    trainer = pl.Trainer(**conf.trainer, **extra_params)
    trainer.fit(model, datamodule=dm)

    valid_metrics = {name: float(mf.compute().item()) for name, mf in model.valid_metrics.items()}
    trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path="best", verbose=False)
    test_metrics = {name: float(mf.compute().item()) for name, mf in model.test_metrics.items()}

    print(", ".join([f"valid_{name}: {v:.4f}" for name, v in valid_metrics.items()]))
    print(", ".join([f" test_{name}: {v:.4f}" for name, v in test_metrics.items()]))

    return {"fold_id": fold_id,
            "model_name": conf.embedding_validation_results.model_name,
            "feature_name": conf.embedding_validation_results.feature_name,
            "scores_valid": valid_metrics,
            "scores_test": test_metrics}


@hydra.main(version_base=None)
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)

    if "seed_everything" in conf:
        pl.seed_everything(conf.seed_everything)

    if conf.data_module.setup.split_by == "embeddings_validation":
        with open(conf.data_module.setup.fold_info, 'r') as f:
            fold_info = json.load(f)
        fold_list = [k for k in sorted(fold_info.keys()) if not k.startswith('_')]
    else:
        raise NotImplementedError("Only embeddings_validation split supported.")

    results = []
    for fold_id in fold_list:
        logger.info(f"==== Fold [{fold_id}/{len(fold_list)}] fit-test start ====")
        result = fold_fit_test(conf, fold_id)
        results.append(result)

    stats_file = conf.embedding_validation_results.output_path
    if stats_file is not None:
        with open(stats_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Embeddings validation scores saved to {stats_file}")

    for name in results[0]["scores_valid"].keys():
        valid_scores = np.array([x["scores_valid"][name]for x in results])
        test_scores = np.array([x["scores_test"][name] for x in results])
        print(f'Valid {name:10}: {valid_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in valid_scores)}], '
              f'Test  {name:10}: {test_scores.mean():.3f} [{", ".join(f"{t:.3f}" for t in test_scores)}]')


if __name__ == "__main__":
    main()
