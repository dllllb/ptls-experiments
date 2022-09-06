import pickle
import numpy as np
from pathlib import Path
import hydra
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window

from ptls.preprocessing import PysparkDataPreprocessor
import logging


FILE_NAME_TRAIN = 'train.csv'
FILE_NAME_TEST = 'test.csv'
COL_EVENT_TIME = 'TRDATETIME'

logger = logging.getLogger(__name__)


def get_df_trx(conf_pp):
    data_path = Path(conf_pp.data_path)
    spark = SparkSession.builder.getOrCreate()

    df_train = spark.read.csv(str(data_path / FILE_NAME_TRAIN), header=True).drop(
        'period', 'term_id', 'target_flag', 'target_sum')
    df_test = spark.read.csv(str(data_path / FILE_NAME_TEST), header=True).drop(
        'period', 'term_id')
    logger.info(f'Loaded {df_train.count()} records from "{FILE_NAME_TRAIN}"')
    logger.info(f'Loaded {df_test.count()} records from "{FILE_NAME_TEST}"')

    df = df_train.union(df_test)

    # event_time mapping
    df = df.withColumn('_et_day', F.substring(F.col(COL_EVENT_TIME), 1, 7))
    df = df.withColumn('_et_day', F.unix_timestamp('_et_day', 'ddMMMyy'))

    df = df.withColumn('_et_time', F.substring(F.col(COL_EVENT_TIME), 9, 8))
    df = df.withColumn('_et_time', F.unix_timestamp('_et_time', 'HH:mm:ss'))

    df = df.withColumn('event_time', F.col('_et_day') + F.col('_et_time'))
    df = df.drop('_et_day', '_et_time', COL_EVENT_TIME)

    for col in df.columns:
        df = df.withColumnRenamed(col, col.lower())

    df = df.withColumn('amount', F.col('amount').cast('float'))

    return df

def get_df_target(conf_pp):
    data_path = Path(conf_pp.data_path)
    spark = SparkSession.builder.getOrCreate()
    df_target = spark.read.csv(str(data_path / FILE_NAME_TRAIN), header=True)
    df_target = df_target.groupby(conf_pp.col_client_id).agg(
        F.first(conf_pp.col_target).cast('int').alias(conf_pp.col_target)
    )
    return df_target


def split_target(conf_pp, df):
    df = df.withColumn('seq_len', F.ceil(F.log(F.col('seq_len')) / np.log(4)).cast('string'))
    df = df.withColumn(
        'hash', F.hash(F.concat(F.col(conf_pp.col_client_id).cast('string'), F.lit(conf_pp.salt))) / 2 ** 32 + 0.5)
    df = df.withColumn(
        'fold_id', F.row_number().over(Window.partitionBy(conf_pp.col_target, 'seq_len').orderBy('hash')) % conf_pp.n_folds) \
        .drop('hash', 'seq_len')
    return df


def split_fold(conf_pp, fold_id, df_target, df_trx):
    spark = SparkSession.builder.getOrCreate()
    folds_path = Path(conf_pp.folds_path)
    preproc = PysparkDataPreprocessor(
        col_id=conf_pp.col_client_id,
        col_event_time='event_time',
        event_time_transformation='none',
        cols_category=["mcc", "channel_type", "currency", "trx_category"],
        cols_numerical=['amount'],
        cols_last_item=[conf_pp.col_target],
    )
    df_train_trx = df_trx.join(df_target, how='left', on=conf_pp.col_client_id).where(
        F.coalesce(F.col('fold_id'), F.lit(-1)) != fold_id).drop('fold_id')
    df_test_trx = df_trx.join(
        df_target.where(F.col('fold_id') == fold_id).drop('fold_id'),
        how='inner', on=conf_pp.col_client_id)
    df_train_data = preproc.fit_transform(df_train_trx)
    df_test_data = preproc.transform(df_test_trx)
    file_name_train = get_file_name_train(fold_id)
    file_name_test = get_file_name_test(fold_id)
    df_train_data.write.parquet(str(folds_path / file_name_train), mode='overwrite')
    df_test_data.write.parquet(str(folds_path / file_name_test), mode='overwrite')
    with open(folds_path / f'preproc_{fold_id}.p', 'wb') as f:
        pickle.dump(preproc, f)
    logger.info(f'Preprocessor[{fold_id}].category_dictionary_sizes={preproc.get_category_dictionary_sizes()}')

    for df_name in [file_name_train, file_name_test]:
        df = spark.read.parquet(str(folds_path / df_name))
        logger.info(f'{df_name:30} {df}')

    for df_name in [file_name_train, file_name_test]:
        df = spark.read.parquet(str(folds_path / df_name))
        r_counts = df.groupby().agg(
            F.sum(F.when(F.col(conf_pp.col_target).isNull(), F.lit(1)).otherwise(F.lit(0))).alias('cnt_none'),
            F.sum(F.when(F.col(conf_pp.col_target) == 0, F.lit(1))).alias('cnt_0'),
            F.sum(F.when(F.col(conf_pp.col_target) == 1, F.lit(1))).alias('cnt_1'),
        ).collect()[0]
        cnt_str = ', '.join([
            f'{r_counts.cnt_none:5d} unlabeled'
            f'{r_counts.cnt_0:5d} - 0 class'
            f'{r_counts.cnt_1:5d} - 1 class'
        ])
        logger.info(f'{df_name:30} {cnt_str}')


def get_file_name_train(fold_id):
    file_name_train = f'df_train_data_{fold_id}.parquet'
    return file_name_train


def get_file_name_test(fold_id):
    file_name_test = f'df_test_data_{fold_id}.parquet'
    return file_name_test


@hydra.main(config_path='conf', config_name='data_preprocessing.yaml', version_base=None)
def main(conf):
    logger.info('Start')

    conf_pp = conf.data_preprocessing

    df_target = get_df_target(conf_pp)
    df_trx = get_df_trx(conf_pp)

    n_folds = conf_pp.n_folds
    df_target = df_target.join(
        df_trx.groupby(conf_pp.col_client_id).count().withColumnRenamed('count', 'seq_len'),
        on=conf_pp.col_client_id, how='inner')  # inner drops clients without transactions
    df_target = split_target(conf_pp, df_target)

    df_target.persist()

    for fold_id in range(n_folds):
        split_fold(conf_pp, fold_id, df_target, df_trx)
    logger.info('Done')


if __name__ == '__main__':
    main()
