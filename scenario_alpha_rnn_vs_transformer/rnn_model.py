import os
import numpy as np
import random
from functools import partial
import torch
import pytorch_lightning as pl
import torchmetrics
from ptls.frames.supervised import SequenceToTarget
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.nn import Head

from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing import SeqLenFilter, FeatureFilter, ToTorch
from ptls.frames.supervised.seq_to_target_dataset import SeqToTargetIterableDataset
from ptls.data_load import IterableChain
from ptls.frames import PtlsDataModule

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pickle
from pyspark.sql import Window


SparkAppName = 'alpha_rnn_model'

spark = SparkSession.builder\
    .appName(SparkAppName)\
    .master("local[*]")\
    .config('spark.driver.memory', '64g')\
    .config('spark.driver.maxResultSize', '16g')\
    .getOrCreate()


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
torch.multiprocessing.set_sharing_strategy('file_system')


train_data = spark.read.parquet("data/train_data")
test_data = spark.read.parquet("data/test_data")
feature_emb_dim = 16

numeric_values={'pre_loans_next_pay_summ': 'identity',
                'pre_loans5': 'identity',
                'pre_loans530': 'identity',
                'pre_loans3060': 'identity',
                'pre_loans6090': 'identity',
                'pre_loans90': 'identity'
}

embeddings={}
for col in train_data.columns:
    if col not in ['id'] + list(numeric_values.keys()):
        distinct_in_col_train = train_data.select(F.col(col)).distinct()
        max_train = distinct_in_col_train.select(F.max(F.col(col))).toPandas().to_numpy().squeeze().tolist()

        distinct_in_col_test = test_data.select(F.col(col)).distinct()
        max_test = distinct_in_col_test.select(F.max(F.col(col))).toPandas().to_numpy().squeeze().tolist()

        in_dim = max(max_train, max_test) + 1
        embeddings[col] = {'in': in_dim, 'out': feature_emb_dim}



trx_encoder_params = dict(
    embeddings_noise=0,
    numeric_values=numeric_values,
    embeddings=embeddings,
    emb_dropout=0.3,
    spatial_dropout=True
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(**trx_encoder_params),
    hidden_size=64,
    type='gru',
    bidir=True,
    reducer='last_max_avg'
)

downstream_model = SequenceToTarget(
    seq_encoder=seq_encoder,
    head=Head(
        input_size=seq_encoder.embedding_size * 6,
        hidden_layers_sizes=[128, 32],
        drop_probs=[0.2, 0],
        use_batch_norm=True,
        objective='classification',
        num_classes=2,
    ),
    loss=torch.nn.NLLLoss(),
    metric_list=torchmetrics.AUROC(num_classes=2, compute_on_step=False),
    pretrained_lr=0.001,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=2000, gamma=1),
)


train_pq_files = ParquetFiles('data/train.parquet/')
valid_pq_files = ParquetFiles('data/valid.parquet/')

train_dataset = ParquetDataset(data_files=train_pq_files.data_files, shuffle_files=True)
valid_dataset = ParquetDataset(data_files=valid_pq_files.data_files, shuffle_files=True)

finetune_dm = PtlsDataModule(
    train_data=SeqToTargetIterableDataset(train_dataset, target_col_name='flag'),
    valid_data=SeqToTargetIterableDataset(valid_dataset, target_col_name='flag'),
    train_num_workers=20,
    train_batch_size=1024,
    valid_batch_size=1024,
)

logger = pl.loggers.TensorBoardLogger(
                save_dir='.',
                name='lightning_logs',
                version='_bidir_RNN_emb_dropout_0.3_no_dropout_gru_head_drop_0.2_max&avg_pool_from_hidden'
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./ckpts/_bidir_RNN_emb_dropout_0.3_no_dropout_gru_head_drop_0.2_max&avg_pool_from_hidden/", save_top_k=40, mode='max', monitor="val_AUROC")

trainer_ft = pl.Trainer(
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else 0,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback],
    logger=logger
)

print(f'logger.version = {trainer_ft.logger.version}')
trainer_ft.fit(downstream_model, finetune_dm)
print(trainer_ft.logged_metrics)
