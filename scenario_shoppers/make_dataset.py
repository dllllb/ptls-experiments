import argparse, datetime, logging, os
import functools, itertools, operator
import numpy as np
import pandas as pd

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

ID_COL = "id"
DATE_COL = "date"
FCAT_COL = "category"
FNUM_COL = "purchasequantity"
FVAL_COL = "purchaseamount"
EVENT_TIME = "event_time"
SEQ_LEN = "seq_len"
TARGET_BIN = "target_bin"
TARGET_DIST = "target_dist"
TARGET_SUM = "target_sum"
TARGET_VAR = "target_var"
TARGET_LOGVAR = "target_logvar"
USECOLS = [ID_COL, DATE_COL, FCAT_COL, FNUM_COL, FVAL_COL]


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


def main(args):
    inp_file = os.path.abspath(args.data)
    target_file = os.path.join(os.path.dirname(inp_file), "train_target.csv")
    assert os.path.exists(inp_file) and not os.path.exists(target_file), "check i/o files existence"

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(f"{inp_file}.log", mode="w")])
    data = pd.read_csv(inp_file, usecols=USECOLS).dropna()
    logger.info(f"Total {data.shape[0]} rows loaded.")

    data.drop(index=data[(data[FNUM_COL] <= 0) | (data[FVAL_COL] <= 0)].index, inplace=True)
    logger.info(f"Total {data.shape[0]} rows after dropping neg.purchases.")

    data[DATE_COL] = data[DATE_COL].apply(str_to_number)
    data[DATE_COL] = (data[DATE_COL] - data[DATE_COL].min()) // args.dt

    data["ppu"] = data[FVAL_COL] / data[FNUM_COL]
    ppu = data.groupby(by=FCAT_COL)["ppu"].mean().reset_index()
    data = data.merge(ppu, on=FCAT_COL, suffixes=["_old", None])
    data[FVAL_COL] = data["ppu"] * data[FNUM_COL]

    vc = data.groupby(by=FCAT_COL)[FVAL_COL].sum().sort_values(ascending=False) / data[FVAL_COL].sum()
    drop_cats = set(vc[vc.cumsum(0) > args.fsum].index)
    data.drop(index=data[data[FCAT_COL].isin(drop_cats)].index, inplace=True)
    data[FCAT_COL], MAX_CAT = encode_col(data[FCAT_COL])
    logger.info(f"Max.category = {MAX_CAT} for total.sum.fraction = {args.fsum}.")

    ppu = data[[FCAT_COL, "ppu"]].groupby(by=FCAT_COL).first().reset_index()
    ppu.to_csv(os.path.join(os.path.dirname(inp_file), "train_cvdict.csv"), header=True, index=False)
    data.drop(columns=["ppu", "ppu_old"], inplace=True)

    group = data.groupby(by=[ID_COL, DATE_COL], as_index=True)
    data = pd.concat([
        group.apply(collapse_seq, [(FCAT_COL, FNUM_COL)]).transform({
            FCAT_COL: operator.itemgetter(0),
            FNUM_COL: operator.itemgetter(1)}),
        group[FVAL_COL].sum()
        ], axis=1, join="inner").reset_index()

    data[SEQ_LEN] = data[FCAT_COL].apply(len)
    logger.info(f"Seqs collapsed within time window = {args.dt} (days).")

    data = data.groupby(by=ID_COL, as_index=True).apply(chain_seq, n_cats=MAX_CAT, n_steps=args.steps)\
        .transform({
            EVENT_TIME: operator.itemgetter(0),
            FCAT_COL: operator.itemgetter(1),
            FNUM_COL: operator.itemgetter(2),
            SEQ_LEN: operator.itemgetter(3),
            TARGET_DIST: operator.itemgetter(4),
            TARGET_VAR: operator.itemgetter(5)
        }).reset_index().dropna()

    if 0 < args.qlim < 1:
        max_tval = data[TARGET_VAR].quantile(args.qlim)
        data.drop(index=data[data[TARGET_VAR] > max_tval].index, inplace=True)
        logger.info(f"Target distribution limited by {max_tval} within {args.qlim} quantile.")

    data[TARGET_LOGVAR] = np.log1p(data[TARGET_VAR])
    data[TARGET_BIN] = pd.qcut(data[TARGET_VAR], q=args.bins, labels=False)
    data[TARGET_SUM] = data[TARGET_DIST].apply(np.sum)
    logger.info(f"Whole dataset {data.shape} constructed.")

    q_vals = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1]
    q_tab = pd.concat([
        data[SEQ_LEN].quantile(q_vals),
        data[TARGET_SUM].quantile(q_vals),
        data[TARGET_VAR].quantile(q_vals)
        ], axis=1)
    logger.info("Quantiles calculated:\n" + repr(q_tab))
    data.drop(columns=[SEQ_LEN], inplace=True)

    mem_total = int(data.memory_usage(deep=True, index=False).sum() / 10 ** 6)
    logger.info(f"Dataset total memory usage (MB): {mem_total}.")
    logger.info(f"Dataset sample:\n" + repr(data.head(10)))

    os.mkdir(os.path.join(os.path.dirname(inp_file), "train"))
    if args.frac == 0:
        out_file = os.path.join(os.path.dirname(inp_file), f"train/{data.shape[0]}.parquet")
        data.to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        logger.info(f"Whole dataset {data.shape} saved to [{out_file}].")
    else:
        rs = np.random.RandomState(args.seed)
        train_mask = rs.rand(data.shape[0])
        train_mask = train_mask > np.percentile(train_mask, 100 * args.frac)

        train_shape = (train_mask.sum(), data.shape[1])
        out_file = os.path.join(os.path.dirname(inp_file), f"train/{train_shape[0]}.parquet")
        data[train_mask].to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        logger.info(f"Train dataset {train_shape} saved to [{out_file}].")

        os.mkdir(os.path.join(os.path.dirname(inp_file), "test"))
        test_shape = (data.shape[0] - train_mask.sum(), data.shape[1])
        out_file = os.path.join(os.path.dirname(inp_file), f"test/{test_shape[0]}.parquet")
        data[~train_mask].to_parquet(out_file, index=False, engine="pyarrow", partition_cols=None)
        data[~train_mask][ID_COL].to_csv(
            os.path.join(os.path.dirname(inp_file), "test_ids.csv"),
            header=True, index=False
        )
        logger.info(f"Test dataset {test_shape} saved to [{out_file}].")

    target_cols = [ID_COL, TARGET_VAR, TARGET_LOGVAR, TARGET_BIN, TARGET_SUM, TARGET_DIST]
    data[target_cols].to_csv(target_file, header=True, index=False)
    logger.info(f"Whole target data ({data.shape[0]}, {len(target_cols)}) saved to [{target_file}].")


def make_sample(args):
    import gzip
    inp_file = os.path.abspath(args.data)
    out_file = inp_file.rsplit(".", 2)[0] + "_sample.csv.gz"
    assert os.path.exists(inp_file) and not os.path.exists(out_file), "check i/o files existence"
    uids = set()
    with gzip.open(inp_file, "rt") as fin, gzip.open(out_file, "wt") as fout:
        header = next(fin)
        uid_index = header.strip().split(",").index(args.id)
        fout.write(header)
        for line in fin:
            uid = line.strip("\n").split(",")[uid_index]
            if len(uids) < args.size:
                uids.add(uid)
            if uid in uids:
                fout.write(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Preprocessor for [kaggle.com/c/acquire-valued-shoppers-challenge] dataset.",
        usage="%(prog)s [options]"
    )
    p.add_argument("--data", type=os.path.abspath, help="abspath to data (required)", metavar="<path>", required=True)
    p.add_argument("--dt", type=int, help="time window for seq.collapse and distrib.target", metavar="[10]", default=10)
    p.add_argument("--fsum", type=float, help="total sum fraction to limit max.category number ", metavar="[1.0]", default=1)
    p.add_argument("--steps", type=int, help="forward steps for (long) target", metavar="[12]", default=12)
    p.add_argument("--bins", type=int, help="bins for target quantization", metavar="[10]", default=10)
    p.add_argument("--frac", type=float, help="test sample fraction", metavar="[0.1]", default=0.1)
    p.add_argument("--qlim", type=float, help="quantile for target distribution up-limit", metavar="[0.0]", default=0)
    p.add_argument("--seed", type=int, help="RNG seed", metavar="[42]", default=42)
    p.add_argument("--size", type=int, help="sample from raw CSV-data if size > 0", metavar="[0]", default=0)
    p.add_argument("--id", type=str, help="column name to count sample", metavar=f"[{ID_COL}]", default=ID_COL)
    args = p.parse_args()

    if args.size > 0:
        make_sample(args)
    else:
        main(args)
