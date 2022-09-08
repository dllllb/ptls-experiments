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


# def get_data_module(conf, fold_id):
#     folds_path = Path(conf.data_preprocessing.folds_path)
#     train_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)))
#     test_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_test(fold_id)))
#
#     train_ds = PersistDataset(ParquetDataset(
#         train_files,
#         i_filters=[
#             iterable_processing.TargetEmptyFilter(target_col=conf.data_preprocessing.col_target),
#             iterable_processing.ISeqLenLimit(max_seq_len=conf.data_module.max_seq_len),
#         ],
#     ))
#     test_ds = PersistDataset(ParquetDataset(
#         test_files,
#         i_filters=[
#             iterable_processing.ISeqLenLimit(max_seq_len=conf.data_module.max_seq_len),
#         ],
#     ))
#
#     if conf.data_module.augmentations is not None:
#         train_ds = AugmentationDataset(
#             train_ds,
#             f_augmentations=instantiate(conf.data_module.augmentations),
#         )
#     datasets = dict(
#         train_data=train_ds,
#         test_data=test_ds,
#     )
#     datasets = {
#         k: SeqToTargetDataset(v, target_col_name=conf.data_preprocessing.col_target, target_dtype='int')
#         for k, v in datasets.items()
#     }
#
#     data_module = PtlsDataModule(
#         **datasets,
#         **conf.data_module.dm_params,
#     )
#     return data_module


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


# def get_pl_module(conf):
#     return instantiate(conf.pl_module)


# def run_supervised_model(conf, fold_id):
#     data_module = get_data_module(conf, fold_id)
#     pl_module = get_pl_module(conf)
#     trainer = pl.Trainer(
#         gpus=1,
#         limit_train_batches=conf.trainer.limit_train_batches,
#         max_epochs=conf.trainer.max_epochs,
#         enable_checkpointing=False,
#         logger=pl.loggers.TensorBoardLogger(
#             save_dir=to_absolute_path(conf.tb_save_dir),
#             name=f'{conf.mode}_fold={fold_id}',
#             version=None,
#             default_hp_metric=False,
#         ),
#         callbacks=[
#             pl.callbacks.LearningRateMonitor(),
#         ],
#     )
#     trainer.fit(pl_module, data_module)
#     logger.info(f'logged_metrics={trainer.logged_metrics}')
#     train_auroc_t = trainer.logged_metrics['train_auroc'].item()
#     train_auroc_v = trainer.logged_metrics['val_auroc'].item()
#     test_metrics = trainer.test(pl_module, data_module, verbose=False)
#     logger.info(f'logged_metrics={trainer.logged_metrics}')
#     logger.info(f'test_metrics={test_metrics}')
#     final_test_metric = test_metrics[0]['test_auroc']
#
#     if conf.mode == 'valid':
#         trainer.logger.log_hyperparams(
#             params=flat_conf(conf),
#             metrics={
#                 f'hp/auroc': final_test_metric,
#                 f'hp/auroc_t': train_auroc_t,
#                 f'hp/auroc_v': train_auroc_v,
#             },
#         )
#
#     logger.info(f'[{conf.mode}] on fold[{fold_id}] finished with {final_test_metric:.4f}')
#     return final_test_metric


def estimate_agg_embeddings(conf, fold_id):
    agg_model = instantiate(conf.agg_model)
    inf_module = InferenceModule(agg_model, model_out_name='emb')

    train_dl, test_dl = get_inference_data(conf, fold_id)

    p_trainer = pl.Trainer(
        gpus=1,
        max_epochs=-1,
        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=to_absolute_path(conf.tb_save_dir),
            name=f'{conf.mode}_fold={fold_id}',
            version=None,
            default_hp_metric=False,
        ),
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

    if conf.mode == 'valid':
        y_predict_train = downstream_model.predict_proba(x_train)[:, 1]
        train_auroc_t = roc_auc_score(y_train, y_predict_train)

        p_trainer.logger.log_hyperparams(
            params=flat_conf(conf),
            metrics={
                f'hp/auroc': final_test_metric,
                f'hp/auroc_t': train_auroc_t,
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
[2022-09-07 10:21:03,190][__main__][INFO] - test done, folds=[1, 2, 3, 4, 5], mean=0.8145, std=0.0076, mean_pm_std=[0.8069, 0.8220], confidence95=[0.8040, 0.8250], values=[0.8263, 0.8077, 0.8145, 0.8053, 0.8185]
[2022-09-07 13:08:08,183][__main__][INFO] - test done, folds=[1, 2, 3, 4, 5], mean=0.8007, std=0.0136, mean_pm_std=[0.7871, 0.8143], confidence95=[0.7818, 0.8196], values=[0.8122, 0.7964, 0.7772, 0.8021, 0.8156]
                                                        agg without amount    mean=0.7966, std=0.0040, mean_pm_std=[0.7925, 0.8006], confidence95=[0.7909, 0.8022],
                                                          random linear       mean=0.8020, std=0.0146, mean_pm_std=[0.7874, 0.8166], confidence95=[0.7818, 0.8222]
'''