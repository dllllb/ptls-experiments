import ast
import hydra
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from collections import OrderedDict
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChunkedDataset:
    def __init__(self, conf, fold_id=0, q=0.995):
        self.device = torch.device(conf.device)
        self.k = conf.pl_module.seq_encoder.trx_encoder.embeddings.category["in"] - 1
        self.q = q

        df = pd.read_csv(os.path.join(conf.work_dir, f"version_{fold_id}/prediction.csv"))
        self.ids = df["seq_id"].values
        self.data = torch.from_numpy(df.values[:, 1:].astype(np.float32)).float().to(self.device)

        self.chunk_size = conf.monte_carlo.chunk
        self.n_chunks, rest = divmod(len(self.ids), self.chunk_size)
        self.n_chunks += rest > 0

        col_target = conf.data_module.setup.col_target
        cw = pd.read_csv(os.path.join(conf.data_path, "train_cvdict.csv"))
        df = pd.read_csv(os.path.join(conf.data_path, "train_target.csv"))
        df[col_target] = df[col_target].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        self.qval = int(np.quantile(df[col_target].apply(lambda x: x.sum()), self.q))
        if self.k < cw.shape[0]:
            w = df[col_target].sum()[self.k - 1:]
            w = np.hstack((np.ones(self.k - 1), w / w.sum())) * cw["ppu"].values
            self.cat_weight = np.hstack((w[:self.k - 1], w[self.k - 1:].sum()))
        else:
            self.cat_weight = cw["ppu"].values
        self.target = df.drop(columns=[col_target])

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        if i < 0 or i >= self.n_chunks:
            raise IndexError("index out of range")
        return self.data[i * self.chunk_size:(i + 1) * self.chunk_size]


class OneStepPredictor:
    def __init__(self, conf, fold_id=0):
        self.device = torch.device(conf.device)
        self.models = ("head", "seq_encoder")
        self.head = hydra.utils.instantiate(conf.pl_module.head)
        self.seq_encoder = hydra.utils.instantiate(conf.pl_module.seq_encoder)

        self.ckpt_path = os.path.join(conf.work_dir, f"version_{fold_id}/checkpoints/epoch={conf.ckpt:02d}.ckpt")
        params = torch.load(self.ckpt_path, map_location=self.device)["state_dict"]
        new_params = {name: OrderedDict() for name in self.models}
        for k, v in params.items():
            pref, new_k = k.split(".", 1)
            if pref in self.models:
                new_params[pref][new_k] = v

        self.seq_encoder.load_state_dict(new_params["seq_encoder"])
        self.seq_encoder.eval().requires_grad_(False).to(self.device)

        self.head.load_state_dict(new_params["head"])
        self.head.eval().requires_grad_(False).to(self.device)

    def __call__(self, x, h0=None):
        with torch.no_grad():
            h = self.seq_encoder(x, h0)
            if h0 is not None:
                h = torch.where((x.seq_lens == 0).unsqueeze(1), h0, h)
            y = self.head(h)
        return y, h


class Sampler:
    def __init__(self, conf):
        self.k = conf.pl_module.seq_encoder.trx_encoder.embeddings.category["in"] - 1
        self.vars = ("category", "purchasequantity")

    def __call__(self, x, num_max=None):
        extra_size = x.shape[1] - self.k
        if extra_size == 0:
            res = torch.poisson(x)
        elif extra_size in (2, 3):
            if extra_size == 2:
                nums, dist = x[:, 0].long(), F.softmax(x[:, 1:], dim=1)
            else:
                nums, dist = (0.5 + torch.exp(x[:, 0])).long(), F.softmax(x[:, 2:], dim=1)
            nums = torch.clamp(nums, min=0, max=num_max)
            res = torch.multinomial(dist, num_samples=nums.max().item(), replacement=True)
            res = self.bincount(res, nums)[:, 1:]
        else:
            raise Exception(f"{self.__class__} got incorrect input sizes")
        out = res.sort(dim=1, descending=True)
        pb = PaddedBatch(payload={self.vars[0]: 1 + out.indices, self.vars[1]: out.values},
                         length=(out.values > 0).sum(dim=1).long())
        return PaddedBatch({k: v * pb.seq_len_mask for k, v in pb.payload.items()}, pb.seq_lens), res

    def bincount(self, x, mask=None):
        if mask is None:
            mask = torch.ones_like(x)
        else:
            z = torch.zeros(x.shape[0], 1 + x.shape[1], dtype=x.dtype, device=x.device)
            mask = 1 - z.scatter(1, mask.unsqueeze(-1), 1).cumsum(dim=1)[:, :-1]
        out = torch.zeros(x.shape[0], 1 + self.k, dtype=x.dtype, device=x.device).scatter_add_(1, x, mask)
        return out


def monte_carlo(chunk, sampler, predictor, n_repeats, n_steps=12, num_max=None):
    res = torch.zeros(chunk.shape[0], sampler.k, dtype=torch.float32, device=predictor.device)
    for _ in tqdm(range(n_repeats)):
        h = None
        x = chunk.clone()
        for k in range(n_steps):
            x, r = sampler(x, num_max)
            x, h = predictor(x, h)
            res += r
    return res / n_repeats


@hydra.main(version_base=None)
def main(conf: DictConfig):
    with open(conf.data_module.setup.fold_info) as fi:
        fold_ids = [int(k) for k in sorted(json.load(fi).keys()) if not k.startswith('_')]

    sampler = Sampler(conf)
    col_id, n_repeats, n_steps = conf.data_module.setup.col_id, conf.monte_carlo.repeats, conf.monte_carlo.steps
    total_res = list()
    for fold_id in fold_ids:
        predictor = OneStepPredictor(conf, fold_id)
        if fold_id == fold_ids[0]:
            logger.info("\n" + repr(predictor.seq_encoder) + "\n" + repr(predictor.head))
        logger.info(f"Model weights restored from: {predictor.ckpt_path}.")
        dataset = ChunkedDataset(conf, fold_id)
        logger.info(f"Dataset {dataset.data.shape} splitted into {dataset.n_chunks} chunks.")
        logger.info(f"Dataset sum(target_dist) {dataset.q}-quantile = {dataset.qval}.")
        res = list()
        for i in range(dataset.n_chunks):
            res.append(monte_carlo(dataset[i], sampler, predictor, n_repeats, n_steps, dataset.qval))
            pred = torch.cat(res, dim=0).cpu().numpy()
            df = {col_id: dataset.ids[:pred.shape[0]], conf.monte_carlo.col: pred.dot(dataset.cat_weight)}
            df.update({f"n_{i}": pred[:, i] for i in range(pred.shape[1])})
            df = pd.DataFrame(df).merge(dataset.target, on=col_id)
            df.to_csv(os.path.join(conf.work_dir, f"monte_carlo_{fold_id}.csv"), header=True, index=False)
            logger.info(f"Done chunk [{1 + i}/{dataset.n_chunks}] for fold [{1 + fold_id}/{len(fold_ids)}].")
        total_res.append(df)
    pd.concat(total_res).to_csv(os.path.join(conf.work_dir, "monte_carlo.csv"), header=True, index=False)


if __name__ == "__main__":
    main()
