import hydra
import json
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def fold_fit_predict(conf, fold_id):
    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model, fold_id=fold_id)
    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(10), ModelCheckpoint(save_top_k=-1, filename="{epoch:02d}")],
        logger=TensorBoardLogger(save_dir=conf.work_dir, name=None), **conf.trainer
    )
    trainer.fit(model, datamodule=dm)
    res = {"fold_id": fold_id}
    for fn in sorted(os.listdir(os.path.join(trainer.log_dir, "checkpoints"))):
        ckpt_path = os.path.join(trainer.log_dir, "checkpoints", fn)
        m = trainer.validate(model=model, dataloaders=dm.val_dataloader(), ckpt_path=ckpt_path)[0]
        if "test_data_path" in conf.data_module.setup.dataset_files:
            m.update(trainer.test(model=model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_path)[0])
        res.update({fn.split(".")[0]: m})
    return res


def fold_predict(conf, fold_id):
    model = hydra.utils.instantiate(conf.pl_module)
    dm = hydra.utils.instantiate(conf.data_module, pl_module=model, fold_id=fold_id)
    dm.prepare_data()
    cwd = os.path.join(conf.work_dir, f"version_{fold_id}")
    pred = pl.Trainer(**conf.trainer, logger=False).predict(
        model=model, dataloaders=dm.predict_dataloader(),
        ckpt_path=os.path.join(cwd, f"checkpoints/epoch={conf.ckpt:02d}.ckpt")
    )
    pd.concat(pred, axis=0).to_csv(os.path.join(cwd, "prediction.csv"), header=True, index=False)


@hydra.main(version_base=None)
def main(conf: DictConfig):
    pl.seed_everything()
    with open(conf.data_module.setup.fold_info) as fi:
        fold_ids = [k for k in sorted(json.load(fi).keys()) if not k.startswith('_')]

    if conf.ckpt is None:
        res = [fold_fit_predict(conf, fold_id) for fold_id in fold_ids]
        eks = sorted(res[0].keys() - {"fold_id"})
        avg = {mk: [np.array([x[ek][mk] for x in res]).mean() for ek in eks] for mk in res[0][eks[0]].keys()}
        with open(os.path.join(conf.work_dir, "cv_result.json"), "w") as fo:
            json.dump([{k: "  ".join([f"{v:.3f}" for v in avg[k]]) for k in avg}] + res, fo, indent=2)
    else:
        for fold_id in fold_ids:
            fold_predict(conf, fold_id)


if __name__ == "__main__":
    main()
