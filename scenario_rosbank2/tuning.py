import logging
from itertools import chain
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import scipy.stats
import torch
from hydra.utils import to_absolute_path, instantiate
from omegaconf import DictConfig, ListConfig
from ptls.data_load import iterable_processing
from ptls.data_load.datasets import parquet_file_scan, ParquetDataset, PersistDataset, AugmentationDataset
from ptls.frames import PtlsDataModule
from ptls.frames.supervised import SequenceToTarget, SeqToTargetDataset
from ptls.data_load.padded_batch import PaddedBatch
from ptls.nn.trx_encoder.batch_norm import RBatchNormWithLens, RBatchNorm
from sklearn.model_selection import train_test_split

from data_preprocessing import get_file_name_train, get_file_name_test

logger = logging.getLogger(__name__)


class SequenceToTargetEx(SequenceToTarget):
    def __init__(self, weight_decay_trx_encoder, **params):
        super().__init__(**params)
        self.hparams['weight_decay_trx_encoder'] = weight_decay_trx_encoder

    def configure_optimizers(self):
        def _is_trx_in(n):
            return n.find('trx_encoder.') >= 0 or n.find('_ih_l0') >= 0

        trx_encoder_parameters = [(n, p) for n, p in self.seq_encoder.named_parameters() if _is_trx_in(n)]
        logger.info(f'Input parameters: {[n for n, p in trx_encoder_parameters]}')
        # logger.info(f'All parameters: {[n for n, p in self.seq_encoder.named_parameters()]}')

        parameters = [
            {'params': chain(
                self.head.parameters(),
                [p for n, p in self.seq_encoder.named_parameters() if not _is_trx_in(n)],
            )},
            {'params': [p for n, p in trx_encoder_parameters], 'weight_decay': self.hparams.weight_decay_trx_encoder},
        ]
        optimizer = self.optimizer_partial(parameters)
        scheduler = self.lr_scheduler_partial(optimizer)
        return [optimizer], [scheduler]


class NumToVector(torch.nn.Module):
    def __init__(self, embeddings_size):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(1, 1, embeddings_size), requires_grad=True)

    def forward(self, x):
        return x * self.w + self.b


class BlockTrxEncoder(torch.nn.Module):
    def __init__(self,
                 embeddings=None,
                 embeddings_size=1,
                 numeric_values=None,
                 use_batch_norm_with_lens=False,
                 ):
        super().__init__()
        self.scalers = torch.nn.ModuleDict()

        self.use_batch_norm_with_lens = use_batch_norm_with_lens
        self.clip_replace_value = 'max'
        self.embeddings_size = embeddings_size

        for name, scaler_name in numeric_values.items():
            self.scalers[name] = torch.nn.Sequential(
                RBatchNormWithLens() if self.use_batch_norm_with_lens else RBatchNorm(),
                NumToVector(embeddings_size),
            )

        self.embeddings = torch.nn.ModuleDict()
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            self.embeddings[emb_name] = torch.nn.Embedding(
                num_embeddings=emb_props['in'],
                embedding_dim=embeddings_size,
                padding_idx=0,
                max_norm=None,
            )

    def forward(self, x: PaddedBatch):
        processed = []

        for field_name, embed_layer in self.embeddings.items():
            feature = self._smart_clip(x.payload[field_name].long(), embed_layer.num_embeddings)
            processed.append(embed_layer(feature))

        for value_name, scaler in self.scalers.items():
            if self.use_batch_norm_with_lens:
                res = scaler(PaddedBatch(x.payload[value_name].float(), x.seq_lens))
            else:
                res = scaler(x.payload[value_name].float())
            processed.append(res)

        out = torch.stack(processed, 2).sum(dim=2)  # BTFH -> BTH
        return PaddedBatch(out, x.seq_lens)

    def _smart_clip(self, values, max_size):
        if self.clip_replace_value == 'max':
            return values.clip(0, max_size - 1)
        else:
            res = values.clone()
            res[values >= max_size] = self.clip_replace_value
            return res

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        return self.embeddings_size

    @property
    def category_names(self):
        """Returns set of used feature names
        """
        return set([field_name for field_name in self.embeddings.keys()] +
                   [value_name for value_name in self.scalers.keys()] +
                   [pos_name for pos_name in self.pos.keys()]
                   )

    @property
    def category_max_size(self):
        """Returns dict with categorical feature names. Value is dictionary size
        """
        return {k: v.num_embeddings for k, v in self.embeddings.items()}


def get_data_module(conf, fold_id):
    folds_path = Path(conf.data_preprocessing.folds_path)
    train_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_train(fold_id)))
    test_files = parquet_file_scan(to_absolute_path(folds_path / get_file_name_test(fold_id)))

    all_train_ds = PersistDataset(ParquetDataset(
        train_files,
        i_filters=[
            iterable_processing.TargetEmptyFilter(target_col=conf.data_preprocessing.col_target),
            iterable_processing.ISeqLenLimit(max_seq_len=conf.data_module.max_seq_len),
        ],
    ))
    # all_train_ix = np.arange(len(all_train_ds))
    # all_train_y = [rec[conf.data_preprocessing.col_target].item() for rec in all_train_ds]
    # train_ix, valid_ix = train_test_split(
    #     all_train_ix, stratify=all_train_y,
    #     test_size=conf.data_module.valid_size, random_state=conf.data_module.valid_split_random_state)
    #
    # train_ds = torch.utils.data.Subset(all_train_ds, train_ix)
    # valid_ds = torch.utils.data.Subset(all_train_ds, valid_ix)

    test_ds = PersistDataset(ParquetDataset(
        test_files,
        i_filters=[
            iterable_processing.ISeqLenLimit(max_seq_len=conf.data_module.max_seq_len),
        ],
    ))

    train_ds = all_train_ds
    if conf.data_module.augmentations is not None:
        train_ds = AugmentationDataset(
            train_ds,
            f_augmentations=instantiate(conf.data_module.augmentations),
        )
    valid_ds = PersistDataset(ParquetDataset(
        test_files,
        i_filters=[
            iterable_processing.ISeqLenLimit(max_seq_len=conf.data_module.max_seq_len),
        ],
    ))
    datasets = dict(
        train_data=train_ds,
        valid_data=valid_ds,
        test_data=test_ds,
    )
    datasets = {
        k: SeqToTargetDataset(v, target_col_name=conf.data_preprocessing.col_target, target_dtype='int')
        for k, v in datasets.items()
    }

    data_module = PtlsDataModule(
        **datasets,
        **conf.data_module.dm_params,
    )
    return data_module


def get_pl_module(conf):
    return instantiate(conf.pl_module)


def run_model(conf, fold_id):
    data_module = get_data_module(conf, fold_id)
    pl_module = get_pl_module(conf)
    trainer = pl.Trainer(
        gpus=1,
        limit_train_batches=conf.trainer.limit_train_batches,
        max_epochs=conf.trainer.max_epochs,
        enable_checkpointing=False,
        logger=pl.loggers.TensorBoardLogger(
            save_dir=to_absolute_path(conf.tb_save_dir),
            name=f'{conf.mode}_fold={fold_id}',
            version=None,
            default_hp_metric=False,
        ),
        callbacks=[
            pl.callbacks.LearningRateMonitor(),
        ],
    )
    trainer.fit(pl_module, data_module)
    logger.info(f'logged_metrics={trainer.logged_metrics}')
    train_auroc_t = trainer.logged_metrics['train_auroc'].item()
    train_auroc_v = trainer.logged_metrics['val_auroc'].item()
    test_metrics = trainer.test(pl_module, data_module, verbose=False)
    logger.info(f'logged_metrics={trainer.logged_metrics}')
    logger.info(f'test_metrics={test_metrics}')
    final_test_metric = test_metrics[0]['test_auroc']

    if conf.mode == 'valid':
        trainer.logger.log_hyperparams(
            params=flat_conf(conf),
            metrics={
                f'hp/auroc': final_test_metric,
                f'hp/auroc_t': train_auroc_t,
                f'hp/auroc_v': train_auroc_v,
            },
        )

    logger.info(f'[{conf.mode}] on fold[{fold_id}] finished with {final_test_metric:.4f}')
    return final_test_metric


def main_valid(conf):
    valid_fold = 0
    result_fold = run_model(conf, valid_fold)
    logger.info('Validation done')
    return result_fold


def main_test(conf):
    test_folds = [i for i in range(1, conf.data_preprocessing.n_folds)]
    results = []
    for fold_id in test_folds:
        result_fold = run_model(conf, fold_id)
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


@hydra.main(config_path='conf', version_base=None, config_name='nn_fit_target_block')
def main(conf):
    mode = conf.mode

    if mode == 'valid':
        return main_valid(conf)
    if mode == 'test':
        return main_test(conf)
    raise AttributeError(f'`conf.mode should be valid or test. Found: {mode}')

if __name__ == '__main__':
    main()

"""
Original:
test done, folds=[1, 2, 3, 4, 5, 6, 7, 8, 9], mean=0.7991, std=0.0211, mean_pm_std=[0.7780, 0.8202], confidence95=[0.7819, 0.8163], values=[0.7940, 0.7964, 0.7961, 0.7812, 0.8072, 0.7908, 0.7760, 0.8535, 0.7967]
test done, folds=[1, 2, 3, 4, 5, 6, 7, 8, 9], mean=0.8028, std=0.0211, mean_pm_std=[0.7817, 0.8239], confidence95=[0.7856, 0.8200], values=[0.7975, 0.7829, 0.7925, 0.7884, 0.8225, 0.7957, 0.7857, 0.8531, 0.8068]
test done, folds=[1, 2, 3, 4, 5, 6, 7, 8, 9], mean=0.7950, std=0.0187, mean_pm_std=[0.7763, 0.8137], confidence95=[0.7798, 0.8102], values=[0.8049, 0.7798, 0.7727, 0.7855, 0.8021, 0.7833, 0.8310, 0.8170, 0.7787]
test done, folds=[1, 2, 3, 4, 5, 6, 7, 8, 9], mean=0.7980, std=0.0183, mean_pm_std=[0.7797, 0.8163], confidence95=[0.7831, 0.8129], values=[0.8028, 0.7685, 0.7871, 0.7906, 0.8218, 0.7950, 0.8144, 0.8243, 0.7778]
test done, folds=[1, 2, 3, 4, 5, 6, 7, 8, 9], mean=0.7981, std=0.0073, mean_pm_std=[0.7908, 0.8054], confidence95=[0.7922, 0.8041], values=[0.7965, 0.7897, 0.7898, 0.8015, 0.7996, 0.7985, 0.7985, 0.8155, 0.7935]

test done, folds=[1, 2, 3, 4, 5], mean=0.8091, std=0.0101, mean_pm_std=[0.7990, 0.8192], confidence95=[0.7951, 0.8231], values=[0.8146, 0.8215, 0.8113, 0.7913, 0.8069]
test done, folds=[1, 2, 3, 4, 5], mean=0.8023, std=0.0056, mean_pm_std=[0.7967, 0.8078], confidence95=[0.7946, 0.8100], values=[0.8064, 0.7992, 0.8027, 0.7936, 0.8095]

test done, folds=[1, 2, 3, 4, 5], mean=0.8133, std=0.0121, mean_pm_std=[0.8012, 0.8254], confidence95=[0.7965, 0.8301], values=[0.8235, 0.8155, 0.8098, 0.7920, 0.8257]


test done, folds=[1, 2, 3, 4, 5], mean=0.8092, std=0.0130, mean_pm_std=[0.7962, 0.8222], confidence95=[0.7911, 0.8272], values=[0.8294, 0.8089, 0.8102, 0.7883, 0.8090]
"""