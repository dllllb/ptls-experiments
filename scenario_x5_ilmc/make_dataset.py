import datetime
import functools
import hydra
import itertools
import logging
import numpy as np
import operator
import os
import pandas as pd
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

DATE_COL = "transaction_datetime"
FCAT_COL = "category"
FNUM_COL = "purchasequantity"
FVAL_COL = "trn_val"
EVENT_TIME = "event_time"
SEQ_LEN = "seq_len"
TARGET_DIST = "target_dist"
TARGET_LOGVAR = "target_logvar"
TARGET_VAR = "target_var"
TARGET_SUM = "target_sum"


@functools.lru_cache(maxsize=None)
def str_to_number(date, denom=86400):
    ts = datetime.datetime.strptime(date.split()[0], "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc).timestamp()
    return float(ts / denom)


def encode_col(col):
    vc = col.value_counts()
    return col.map({k: i + 1 for i, k in enumerate(vc.index)})


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


@hydra.main(version_base=None)
def main(conf: DictConfig):
    col_id = conf.data_module.setup.col_id
    target_file = os.path.join(conf.data_path, "train_target.csv")
    assert os.path.exists(conf.raw_data) and not os.path.exists(target_file), "check i/o files existence"
    os.makedirs(os.path.join(conf.data_path, "train"), exist_ok=True)

    AUXCOL = ["product_id", "transaction_id", "purchase_sum", "trn_sum_from_iss", "trn_sum_from_red"]
    aux_data = pd.read_csv(conf.aux_data, usecols=["product_id", "level_4"]).dropna()
    data = pd.read_csv(conf.raw_data, usecols=AUXCOL+[col_id, DATE_COL, "product_quantity"])
    data = data.merge(aux_data, on="product_id")
    data.rename(columns={"product_quantity": FNUM_COL, "level_4": FCAT_COL}, inplace=True)
    logger.info(f"Total {data.shape[0]} rows loaded.")

    data["purchase_sum"] = data["purchase_sum"].apply(np.round)
    data[FVAL_COL] = (data["trn_sum_from_iss"].fillna(0) + data["trn_sum_from_red"].fillna(0)).apply(np.round)
    data = data.merge(data.groupby(by="transaction_id")[FVAL_COL].sum(), on="transaction_id", suffixes=[None, "_sum"])
    data = data[(data["trn_val_sum"] - 1 <= data["purchase_sum"]) & (data["purchase_sum"] <= data["trn_val_sum"] + 1)]
    data.drop(columns=AUXCOL+["trn_val_sum"], inplace=True)
    data.drop(index=data[(data[FNUM_COL] <= 0) | (data[FVAL_COL] <= 0)].index, inplace=True)
    logger.info(f"Total {data.shape[0]} rows after dropping neg.purchases.")

    data[DATE_COL] = data[DATE_COL].apply(str_to_number)
    data[DATE_COL] = (data[DATE_COL] - data[DATE_COL].min()) // conf.monte_carlo.agg_time

    data["ppu"] = data[FVAL_COL] / data[FNUM_COL]
    ppu = data.groupby(by=FCAT_COL)["ppu"].mean().reset_index()
    data = data.merge(ppu, on=FCAT_COL, suffixes=["_old", None])
    data[FVAL_COL] = data["ppu"] * data[FNUM_COL]

    vc = data.groupby(by=FCAT_COL)[FVAL_COL].sum().sort_values(ascending=False) / data[FVAL_COL].sum()
    data.drop(index=data[data[FCAT_COL].isin(set(vc.index[conf.ncats:]))].index, inplace=True)
    data[FCAT_COL] = encode_col(data[FCAT_COL])
    pc = int(100 * vc.cumsum(0)[conf.ncats])
    logger.info(f"Max.category after limiting: {vc.shape[0]} ==> {conf.ncats} (sum.value = {pc}%).")

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

    data = data.groupby(by=col_id, as_index=True).apply(chain_seq, n_cats=conf.ncats, n_steps=conf.monte_carlo.steps)\
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
    data[TARGET_SUM] = data[TARGET_DIST].apply(np.sum)
    logger.info(f"Whole dataset {data.shape} constructed.")

    q_vals = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.995, 1]
    q_tab = pd.concat([
        data[SEQ_LEN].quantile(q_vals),
        data[TARGET_SUM].quantile(q_vals),
        data[TARGET_VAR].quantile(q_vals),
        data[TARGET_LOGVAR].quantile(q_vals)
        ], axis=1)
    logger.info("Quantiles calculated:\n" + repr(q_tab))
    data.drop(columns=[SEQ_LEN], inplace=True)

    mem_total = int(data.memory_usage(deep=True, index=False).sum() / 10 ** 6)
    logger.info(f"Dataset total memory usage (MB): {mem_total}.")
    logger.info(f"Dataset sample:\n" + repr(data.head(10)))

    out_file = os.path.join(conf.data_path, f"train/{data.shape[0]}.parquet")
    data.to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
    logger.info(f"Whole dataset {data.shape} saved to [{out_file}].")

    target_cols = [col_id, TARGET_VAR, TARGET_LOGVAR, TARGET_SUM, TARGET_DIST]
    data[target_cols].to_csv(target_file, header=True, index=False)
    logger.info(f"Whole target data ({data.shape[0]}, {len(target_cols)}) saved to [{target_file}].")


if __name__ == "__main__":
    main()
