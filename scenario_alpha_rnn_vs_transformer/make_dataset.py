from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pickle
from pyspark.sql import Window

SparkAppName = 'alpha_rnn_vs_transformer'

spark = SparkSession.builder\
    .appName(SparkAppName)\
    .master("local[*]")\
    .config('spark.driver.memory', '64g')\
    .config('spark.driver.maxResultSize', '16g')\
    .getOrCreate()

train_data = spark.read.parquet("data/train_data")
train_target = spark.read.csv("data/train_target.csv", header=True)
train_target = train_target.withColumn('flag', train_target['flag'].cast('int'))
test_data = spark.read.parquet("data/test_data")
test_target = spark.read.csv("data/test_target.csv", header=True)


min_1_cols = ['id', 'rn', 'enc_paym_24', 'enc_paym_20',
              'enc_paym_11', 'pre_loans90', 'pre_loans_outstanding']
for column in train_data.columns:
    if column not in min_1_cols:
        train_data = train_data.withColumn(column, F.col(column) + F.lit(1))
        test_data = test_data.withColumn(column, F.col(column) + F.lit(1))

full_train_df = train_data.groupBy('id')\
    .agg(*[F.collect_list(col).alias(col) for col in train_data.columns if col != 'id'])\
    .join(train_target, ['id']).withColumn('rn', F.reverse('rn')).cache()

test_df = test_data.groupBy('id')\
    .agg(*[F.collect_list(col).alias(col) for col in test_data.columns if col != 'id'])\
    .withColumn('rn', F.reverse('rn'))

valid_df = full_train_df.sample(0.01)
train_df = full_train_df.join(valid_df, full_train_df.id == valid_df.id, how='left_anti')

full_train_df.write.parquet("data/full_train.parquet")
train_df.write.parquet("data/train.parquet")
valid_df.write.parquet("data/valid.parquet")
test_df.write.parquet("data/test.parquet")

