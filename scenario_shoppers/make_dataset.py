import datetime
import functools
import hydra
import itertools
import logging
import numpy as np
import operator
import os
import pandas as pd
import sys
from omegaconf import DictConfig

numexpr_max_threads = int(os.environ.get("NUMEXPR_MAX_THREADS", 0))
if not numexpr_max_threads:
    try:
        import numexpr
        numexpr_max_threads = numexpr.detect_number_of_cores()
        numexpr.set_num_threads(numexpr_max_threads)
    except:
        numexpr_max_threads = "not initialized"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 24)
pd.set_option("display.width", None)
logger = logging.getLogger(__name__)

DATE_COL = "date"
FCAT_COL = "category"
FNUM_COL = "purchasequantity"
FVAL_COL = "purchaseamount"
EVENT_TIME = "event_time"
SEQ_LEN = "seq_len"
TARGET_BIN = "target_bin"
TARGET_DIST = "target_dist"
TARGET_VAR = "target_var"
TARGET_LOGVAR = "target_logvar"


@functools.lru_cache(maxsize=None)
def str_to_number(date, denom=86400):
    ts = datetime.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    return float(ts / denom)


@functools.lru_cache(maxsize=None)
def str_to_datetime(date, utc=False):
    return pd.to_datetime(date, format="%Y-%m-%d", utc=utc)


def encode_col(col):
    vc = col.value_counts()
    return col.map({k: i + 1 for i, k in enumerate(vc.index)}), vc.shape[0]


def collapse_seq(df, pairs):
    res = []
    for col_k, col_v in pairs:
        if col_v is None:
            uniq, cnt = np.unique(df[col_k], return_counts=True)
            res.append(uniq[cnt.argmax()])
        else:
            agg = dict()
            for k, v in zip(df[col_k], df[col_v]):
                agg[k] = agg.get(k, 0) + v
            keys, vals = zip(*agg.items())
            res.extend([list(keys), list(vals)])
    return res


def chain_seq(group, n_cats, n_steps=12):
    if group[DATE_COL].iloc[-1] - group[DATE_COL].iloc[0] < n_steps:
        return 6 * [None]

    end_t = group[DATE_COL].iloc[-1] - n_steps
    target = group[group[DATE_COL] > end_t][FVAL_COL].sum()

    g = group[(group[DATE_COL] > end_t) & (group[DATE_COL] <= end_t + 1)]
    dist = np.zeros(n_cats)
    for k, v in zip(itertools.chain.from_iterable(g[FCAT_COL]),
                    itertools.chain.from_iterable(g[FNUM_COL])):
        dist[k - 1] += v

    g = group[group[DATE_COL] <= end_t]
    return (list(itertools.chain.from_iterable(v * [k] for k, v in zip(g[DATE_COL], g[SEQ_LEN]))),
            list(itertools.chain.from_iterable(g[FCAT_COL])),
            list(itertools.chain.from_iterable(g[FNUM_COL])),
            g[SEQ_LEN].sum(), dist.tolist(), target)


def forward_window(group, event_time, target, dt):
    if (group[event_time].iloc[-1] - group[event_time].iloc[0]).days < dt:
        return pd.Series(data=None, dtype=np.float64, index=group.index)
    ser = group[[event_time, target]].iloc[::-1].rolling(window=f"{1 + dt}D", on=event_time)[target].sum().iloc[::-1]
    end_t = group[event_time].iloc[-1] - pd.Timedelta(days=dt)
    return ser.subtract(group[target]).where(group[event_time] <= end_t, None)


def make_sample(inp_file, col_id, chunk):
    import gzip
    out_file = inp_file.rsplit(".", 2)[0] + "_sample.csv.gz"
    assert os.path.exists(inp_file) and not os.path.exists(out_file), "check i/o files existence"
    uids = set()
    with gzip.open(inp_file, "rt") as fin, gzip.open(out_file, "wt") as fout:
        header = next(fin)
        uid_index = header.strip().split(",").index(col_id)
        fout.write(header)
        for line in fin:
            uid = line.strip("\n").split(",")[uid_index]
            if len(uids) < chunk:
                uids.add(uid)
            if uid in uids:
                fout.write(line)


@hydra.main(version_base=None)
def main(conf: DictConfig):
    col_id = conf.data_module.setup.col_id
    if conf.monte_carlo.agg_time is None:
        make_sample(conf.raw_data, col_id, conf.monte_carlo.chunk)
        sys.exit(0)

    target_file = os.path.join(conf.data_path, "train_target.csv")
    assert os.path.exists(conf.raw_data) and not os.path.exists(target_file), "check i/o files existence"
    os.makedirs(os.path.join(conf.data_path, "train"), exist_ok=True)

    data = pd.read_csv(conf.raw_data, usecols=[col_id, DATE_COL, FCAT_COL, FNUM_COL, FVAL_COL]).dropna()
    logger.info(f"Total {data.shape[0]} rows loaded.")

    data.drop(index=data[(data[FNUM_COL] <= 0) | (data[FVAL_COL] <= 0)].index, inplace=True)
    logger.info(f"Total {data.shape[0]} rows after dropping neg.purchases.")

    data[DATE_COL] = data[DATE_COL].apply(str_to_number)
    data[DATE_COL] = (data[DATE_COL] - data[DATE_COL].min()) // conf.monte_carlo.agg_time

    data["ppu"] = data[FVAL_COL] / data[FNUM_COL]
    ppu = data.groupby(by=FCAT_COL)["ppu"].mean().reset_index()
    data = data.merge(ppu, on=FCAT_COL, suffixes=["_old", None])
    data[FVAL_COL] = data["ppu"] * data[FNUM_COL]

    vc = data.groupby(by=FCAT_COL)[FVAL_COL].sum().sort_values(ascending=False) / data[FVAL_COL].sum()
    drop_cats = set(vc[vc.cumsum(0) > conf.fsum].index) if conf.fsum <= 1 else set(vc.index[int(conf.fsum):])
    data.drop(index=data[data[FCAT_COL].isin(drop_cats)].index, inplace=True)
    data[FCAT_COL], MAX_CAT = encode_col(data[FCAT_COL])
    logger.info(f"Max.category after limiting: {vc.shape[0]} ==> {MAX_CAT}.")

    ppu = data[[FCAT_COL, "ppu"]].groupby(by=FCAT_COL).first().reset_index()
    ppu.to_csv(os.path.join(conf.data_path, "train_cvdict.csv"), header=True, index=False)
    data.drop(columns=["ppu", "ppu_old"], inplace=True)

    group = data.groupby(by=[col_id, DATE_COL], as_index=True)
    data = pd.concat([
        group.apply(collapse_seq, [(FCAT_COL, FNUM_COL)]).transform({
            FCAT_COL: operator.itemgetter(0),
            FNUM_COL: operator.itemgetter(1)}),
        group[FVAL_COL].sum()
        ], axis=1, join="inner").reset_index()

    data[SEQ_LEN] = data[FCAT_COL].apply(len)
    logger.info(f"Seqs collapsed within time window = {conf.monte_carlo.agg_time} (days).")

    data = data.groupby(by=col_id, as_index=True).apply(chain_seq, n_cats=MAX_CAT, n_steps=conf.monte_carlo.steps)\
        .transform({
            EVENT_TIME: operator.itemgetter(0),
            FCAT_COL: operator.itemgetter(1),
            FNUM_COL: operator.itemgetter(2),
            SEQ_LEN: operator.itemgetter(3),
            TARGET_DIST: operator.itemgetter(4),
            TARGET_VAR: operator.itemgetter(5)
        }).reset_index().dropna()

    if 0 < conf.qlim < 1:
        max_tval = data[TARGET_VAR].quantile(conf.qlim)
        data.drop(index=data[data[TARGET_VAR] > max_tval].index, inplace=True)
        logger.info(f"Target distribution limited by {max_tval} within {conf.qlim} quantile.")

    data[TARGET_LOGVAR] = np.log1p(data[TARGET_VAR])
    data[TARGET_BIN] = pd.qcut(data[TARGET_VAR], q=conf.qbin, labels=False)
    logger.info(f"Whole dataset {data.shape} constructed.")

    q_vals = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]
    q_tab = pd.concat([
        data[SEQ_LEN].quantile(q_vals),
        data[TARGET_VAR].quantile(q_vals)
        ], axis=1)
    logger.info("Quantiles calculated:\n" + repr(q_tab))
    data.drop(columns=[SEQ_LEN], inplace=True)

    mem_total = int(data.memory_usage(deep=True, index=False).sum() / 10 ** 6)
    logger.info(f"Dataset total memory usage (MB): {mem_total}.")
    logger.info(f"Dataset sample:\n" + repr(data.head(10)))

    if conf.test_fraction == 0:
        out_file = os.path.join(conf.data_path, f"train/{data.shape[0]}.parquet")
        data.to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        logger.info(f"Whole dataset {data.shape} saved to [{out_file}].")
    else:
        train_mask = np.random.rand(data.shape[0])
        train_mask = train_mask > np.percentile(train_mask, 100 * conf.test_fraction)

        train_shape = (train_mask.sum(), data.shape[1])
        out_file = os.path.join(conf.data_path, f"train/{train_shape[0]}.parquet")
        data[train_mask].to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        logger.info(f"Train dataset {train_shape} saved to [{out_file}].")

        os.makedirs(os.path.join(conf.data_path, "test"), exist_ok=True)
        test_shape = (data.shape[0] - train_mask.sum(), data.shape[1])
        out_file = os.path.join(conf.data_path, f"test/{test_shape[0]}.parquet")
        data[~train_mask].to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        data[~train_mask][col_id].to_csv(
            os.path.join(conf.data_path, "test_ids.csv"),
            header=True, index=False
        )
        logger.info(f"Test dataset {test_shape} saved to [{out_file}].")

    target_cols = [col_id, TARGET_VAR, TARGET_LOGVAR, TARGET_BIN, TARGET_DIST]
    data[target_cols].to_csv(target_file, header=True, index=False)
    logger.info(f"Whole target data ({data.shape[0]}, {len(target_cols)}) saved to [{target_file}].")


if __name__ == "__main__":
    main()
