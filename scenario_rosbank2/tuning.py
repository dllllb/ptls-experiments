import logging
from pathlib import Path
import pandas as pd

import hydra
import numpy as np
import pytorch_lightning as pl
import scipy.stats
import torch
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from ptls.data_load import iterable_processing
from ptls.data_load.datasets import parquet_file_scan, ParquetDataset, PersistDataset, AugmentationDataset
from ptls.frames import PtlsDataModule
from ptls.frames.supervised import SeqToTargetDataset
from ptls.frames.inference_module import InferenceModule
from ptls.data_load.utils import collate_feature_dict
from sklearn.metrics import roc_auc_score

from data_preprocessing import get_file_name_train, get_file_name_test

logger = logging.getLogger(__name__)


def get_pretrain_data_module(conf, fold_id):
    folds_path = Path(conf.data_preprocessing.folds_path)
    train_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)),
                                    valid_rate=0.05, return_part='train')
    valid_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)),
                                    valid_rate=0.05, return_part='valid')

    train_ds = PersistDataset(ParquetDataset(
        train_files,
        i_filters=[
            iterable_processing.SeqLenFilter(min_seq_len=conf.pretrain_data.min_seq_len)
        ],
    ))
    valid_ds = PersistDataset(ParquetDataset(
        valid_files,
        i_filters=[
            iterable_processing.SeqLenFilter(min_seq_len=conf.pretrain_data.min_seq_len)
        ],
    ))

    if conf.pretrain_data.augmentations is not None:
        train_ds = AugmentationDataset(
            train_ds,
            f_augmentations=instantiate(conf.pretrain_data.augmentations),
        )

    frame_dataset_partial = instantiate(conf.pretrain_data.frame_dataset_partial)

    pretrain_data_module = PtlsDataModule(
        train_data=frame_dataset_partial(data=train_ds),
        valid_data=frame_dataset_partial(data=valid_ds),
        **conf.pretrain_data_module,
    )
    return pretrain_data_module


def get_inference_data(conf, fold_id):
    folds_path = Path(conf.data_preprocessing.folds_path)
    train_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)))
    test_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_test(fold_id)))

    # train_ds = ParquetDataset(train_files)
    # values = []
    # for rec in train_ds:
    #     values.append(rec['amount'])
    # values = torch.cat(values, dim=0)
    # bins = torch.quantile(values, torch.linspace(0.0, 1.0, 16))
    #
    # def make_bins(x, bins):
    #     x['amount_bins'] = torch.bucketize(x['amount'], bins) + 1
    #     return x

    train_ds = ParquetDataset(
        train_files,
        i_filters=[
            iterable_processing.TargetEmptyFilter(target_col=conf.data_preprocessing.col_target),
            iterable_processing.ISeqLenLimit(max_seq_len=conf.data_preprocessing.max_seq_len),
            # lambda x: (make_bins(rec, bins) for rec in x),
        ],
    )
    test_ds = ParquetDataset(
        test_files,
        i_filters=[
            iterable_processing.ISeqLenLimit(max_seq_len=conf.data_preprocessing.max_seq_len),
            # lambda x: (make_bins(rec, bins) for rec in x),
        ],
    )
    dl_params = dict(shuffle=False, collate_fn=collate_feature_dict, **conf.dataloader_inference)
    train_dl = torch.utils.data.DataLoader(train_ds, **dl_params)
    test_dl = torch.utils.data.DataLoader(test_ds, **dl_params)
    return train_dl, test_dl


def estimate_frame_seq_embeddings(conf, fold_id):
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=to_absolute_path(conf.tb_save_dir),
        name=f'{conf.mode}_fold={fold_id}',
        version=None,
        default_hp_metric=False,
    )

    pl_module = instantiate(conf.pl_module)
    pretrain_data_module = get_pretrain_data_module(conf, fold_id)

    pretrain_trainer = pl.Trainer(
        gpus=1,
        logger=tb_logger,
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
        ],
        **conf.pretrain_trainer,
    )
    logger.info('Pretrain start')
    pretrain_trainer.fit(pl_module, pretrain_data_module)
    logger.info('Pretrain done')

    seq_encoder = pl_module.seq_encoder
    seq_encoder.is_reduce_sequence = True
    return estimate_embeddings_with_downstream(
        model=seq_encoder,
        tb_logger=tb_logger,
        conf=conf,
        fold_id=fold_id,
        extra_metrics={'hp/recall_top_k': pretrain_trainer.logged_metrics['recall_top_k'].item()},
    )


def estimate_agg_embeddings(conf, fold_id):
    agg_model = instantiate(conf.agg_model)
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=to_absolute_path(conf.tb_save_dir),
        name=f'{conf.mode}_fold={fold_id}',
        version=None,
        default_hp_metric=False,
    )
    return estimate_embeddings_with_downstream(
        model=agg_model,
        tb_logger=tb_logger,
        conf=conf,
        fold_id=fold_id,
        extra_metrics={},
    )


def estimate_embeddings_with_downstream(model, tb_logger, conf, fold_id, extra_metrics):
    inf_module = InferenceModule(model, model_out_name='emb')

    train_dl, test_dl = get_inference_data(conf, fold_id)

    p_trainer = pl.Trainer(
        gpus=1,
        max_epochs=-1,
        enable_progress_bar=True,
        logger=tb_logger,
    )
    df_train = p_trainer.predict(inf_module, train_dl)
    df_test = p_trainer.predict(inf_module, test_dl)

    df_train = pd.concat(df_train, axis=0).set_index(conf.data_preprocessing.col_client_id)
    df_test = pd.concat(df_test, axis=0).set_index(conf.data_preprocessing.col_client_id)

    y_train = df_train[conf.data_preprocessing.col_target]
    y_test = df_test[conf.data_preprocessing.col_target]
    x_train =  df_train.drop(columns=[conf.data_preprocessing.col_target])
    x_test = df_test.drop(columns=[conf.data_preprocessing.col_target])

    logger.info(f'Prepared embeddings. X_train: {x_train.shape}, X_test: {x_test.shape}. '
                f'y_train: {y_train.shape}, y_test: {y_test.shape}')

    downstream_model = instantiate(conf.downstream_model)
    downstream_model.fit(x_train, y_train)
    logger.info(f'Downstream_model fitted')

    y_predict = downstream_model.predict_proba(x_test)[:, 1]
    final_test_metric = roc_auc_score(y_test, y_predict)

    # if conf.mode == 'valid':
    y_predict_train = downstream_model.predict_proba(x_train)[:, 1]
    train_auroc_t = roc_auc_score(y_train, y_predict_train)

    tb_logger.log_hyperparams(
        params=flat_conf(conf),
        metrics={
            f'hp/auroc': final_test_metric,
            f'hp/auroc_t': train_auroc_t,
            **extra_metrics,
        },
    )

    logger.info(f'[{conf.mode}] on fold[{fold_id}] finished with {final_test_metric:.4f}')
    return final_test_metric


def main_valid(conf):
    run_model_partial = instantiate(conf.run_model_partial)
    valid_fold = 0
    result_fold = run_model_partial(conf, valid_fold)
    logger.info('Validation done')
    return result_fold


def main_test(conf):
    run_model_partial = instantiate(conf.run_model_partial)
    test_folds = [i for i in range(1, conf.data_preprocessing.n_folds)]
    results = []
    for fold_id in test_folds:
        result_fold = run_model_partial(conf, fold_id)
        results.append(result_fold)
    results = np.array(results)

    log_resuts(conf, test_folds, results)

    return results.mean()


def flat_conf(conf):
    def _explore():
        for param_name, element in conf.items():
            for k, v in _explore_recursive(param_name, element):
                yield k, v
        yield 'hydra.cwd', Path.cwd()
        yield 'hydra.reuse_cmd', f'--config-dir={Path.cwd()} +conf_override@=config'

    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    for k, v in _explore_recursive(f'{parent_name}.{k}', v):
                        yield k, v
                else:
                    yield f'{parent_name}.{k}', v
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                yield f'{parent_name}.{i}', v
        else:
            yield parent_name, element

    return dict(_explore())


def log_resuts(conf, fold_list, results, float_precision='{:.4f}'):
    def t_interval(x, p=0.95):
        eps = 1e-9
        n = len(x)
        s = x.std(ddof=1)

        return scipy.stats.t.interval(p, n - 1, loc=x.mean(), scale=(s + eps) / (n ** 0.5))

    mean = results.mean()
    std = results.std()
    t_int = t_interval(results)

    results_str = ', '.join([float_precision.format(r) for r in results])
    logger.info(', '.join([
        f'{conf.mode} done',
        f'folds={fold_list}',
        f'mean={float_precision.format(mean)}',
        f'std={float_precision.format(std)}',
        f'mean_pm_std=[{float_precision.format(mean - std)}, {float_precision.format(mean + std)}]',
        f'confidence95=[{float_precision.format(t_int[0])}, {float_precision.format(t_int[1])}]',
        f'values=[{results_str}]',
    ]))

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=to_absolute_path(conf.tb_save_dir),
        name=f'{conf.mode}_mean',
        version=None,
        prefix='',
        default_hp_metric=False,
    )
    tb_logger.log_hyperparams(
        params=flat_conf(conf),
        metrics={f'{conf.mode}_auroc_mean': mean},
    )
    logger.info(f'Results are logged to tensorboard as {tb_logger.name}/{tb_logger.version}')
    logger.info(f'Output logged to "{Path.cwd()}"')


@hydra.main(config_path='conf', version_base=None)
def main(conf):
    # save config for future overrides
    conf_override_path = Path.cwd() / 'conf_override'
    conf_override_path.mkdir()
    OmegaConf.save(config=conf, f=conf_override_path / 'config.yaml')

    mode = conf.mode

    if mode == 'valid':
        return main_valid(conf)
    if mode == 'test':
        return main_test(conf)
    raise AttributeError(f'`conf.mode should be valid or test. Found: {mode}')

if __name__ == '__main__':
    main()

'''
agg_embeddings              mean=0.8145, std=0.0076, mean_pm_std=[0.8069, 0.8220], confidence95=[0.8040, 0.8250], values=[0.8263, 0.8077, 0.8145, 0.8053, 0.8185]
random_trx_pool_embeddings  mean=0.8049, std=0.0104, mean_pm_std=[0.7945, 0.8153], confidence95=[0.7904, 0.8193], values=[0.8225, 0.8052, 0.7938, 0.7949, 0.8080]

Lgbm:
coles_base                  mean=0.8247, std=0.0103, mean_pm_std=[0.8143, 0.8350], confidence95=[0.8103, 0.8390], values=[0.8407, 0.8194, 0.8270, 0.8092, 0.8272]
coles sup     w=1.0         mean=0.8215, std=0.0104, mean_pm_std=[0.8111, 0.8319], confidence95=[0.8071, 0.8359]
coles sup     w=1.0 margin  mean=0.8276, std=0.0097, mean_pm_std=[0.8179, 0.8373], confidence95=[0.8141, 0.8411]


linear                      mean=0.8098, std=0.0095, mean_pm_std=[0.8003, 0.8192], confidence95=[0.7966, 0.8229],

hidden=512:
coles_base                  mean=0.8092, std=0.0072, mean_pm_std=[0.8019, 0.8164], confidence95=[0.7991, 0.8192]
coles_sup                   mean=0.8065, std=0.0092, mean_pm_std=[0.7973, 0.8157], confidence95=[0.7938, 0.8193]

hidden=256:
coles_base                  mean=0.8075, std=0.0073, mean_pm_std=[0.8002, 0.8148], confidence95=[0.7974, 0.8177]
coles_sup                   mean=0.8061, std=0.0085, mean_pm_std=[0.7976, 0.8145], confidence95=[0.7943, 0.8178]

hidden=128:
coles_base                  mean=0.8056, std=0.0073, mean_pm_std=[0.7984, 0.8129], confidence95=[0.7956, 0.8157]   
coles_sup                   mean=0.8060, std=0.0087, mean_pm_std=[0.7973, 0.8147], confidence95=[0.7939, 0.8181]

hidden=64:
coles_base                  mean=0.7981, std=0.0091, mean_pm_std=[0.7889, 0.8072], confidence95=[0.7854, 0.8107],
coles_sup                   mean=0.7996, std=0.0113, mean_pm_std=[0.7883, 0.8109], confidence95=[0.7840, 0.8153]


coles sup     w=1.0 with class center running average
# попробовать вместе с class center running average

'''
