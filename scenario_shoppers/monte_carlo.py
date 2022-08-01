import ast, hydra, logging, os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from collections import OrderedDict
from glob import glob
from omegaconf import DictConfig
from ptls.data_load import PaddedBatch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChunkedDataset:
    def __init__(self, conf):
        self.device = torch.device(conf.device)
        self.k = conf.mc.states

        df = pd.concat([pd.read_csv(fn) for fn in glob(os.path.join(conf.mc.path, "version_*/test_prediction.csv"))])
        df = df.groupby("seq_id").mean().reset_index()
        self.ids = df["seq_id"].values
        self.data = torch.from_numpy(df.values[:, 1:]).float().to(self.device)

        self.chunk_size = conf.mc.chunk
        self.n_chunks, rest = divmod(len(self.ids), self.chunk_size)
        self.n_chunks += rest > 0

        cw = pd.read_csv(os.path.join(conf.mc.target, "train_cvdict.csv"))
        df = pd.read_csv(os.path.join(conf.mc.target, "train_target.csv"))
        if self.k < cw.shape[0]:
            df["target_dist"] = df["target_dist"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
            w = df["target_dist"].sum()[self.k - 1:]
            w = np.hstack((np.ones(self.k - 1), w / w.sum())) * cw["ppu"].values
            self.cat_weight = np.hstack((w[:self.k - 1], w[self.k - 1:].sum()))
        else:
            self.cat_weight = cw["ppu"].values
        self.target = df.drop(columns=["target_sum","target_dist"])

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, i):
        if i < 0 or i >= self.n_chunks:
            raise IndexError("index out of range")
        return self.data[i * self.chunk_size:(i + 1) * self.chunk_size]


class OneStepPredictor:
    def __init__(self, conf):
        self.device = torch.device(conf.device)
        self.models = ("head", "seq_encoder")
        self.head = hydra.utils.instantiate(conf.head)
        self.seq_encoder = hydra.utils.instantiate(conf.seq_encoder)

        if conf.mc.version == "best":
            raise NotImplementedError()
        self.ckpt = sorted(glob(os.path.join(conf.mc.path, "version_*/checkpoints/*.ckpt")))[conf.mc.version]
        params = torch.load(self.ckpt, map_location=self.device)["state_dict"]
        new_params = {name: OrderedDict() for name in self.models}
        for k, v in params.items():
            pref, new_k = k.split(".", 1)
            if pref in self.models:
                new_params[pref][new_k] = v

        self.seq_encoder.load_state_dict(new_params["seq_encoder"])
        self.seq_encoder.eval().requires_grad_(False).to(self.device)

        self.head.load_state_dict(new_params["head"])
        self.head.eval().requires_grad_(False).to(self.device)

    def __call__(self, x, h=None):
        with torch.no_grad():
            h = self.seq_encoder(x, h)
            y = self.head(h)
        return y, h


class Sampler:
    def __init__(self, conf):
        self.k = conf.mc.states
        self.rng = torch.Generator(conf.device)
        self.rng.manual_seed(conf.rng_seed)
        self.vars = ("category", "purchasequantity")

    def __call__(self, x):
        extra_size = x.shape[1] - self.k
        if extra_size == 0:
            res = torch.poisson(x, generator=self.rng)
        elif extra_size in (2, 3):
            if extra_size == 2:
                nums, dist = x[:, 0].long(), F.softmax(x[:, 1:])
            else:
                nums, dist = (0.5 + torch.exp(x[:, 0])).long(), F.softmax(x[:, 2:])
            res = torch.multinomial(dist, num_samples=nums.max().item(), replacement=True, generator=self.rng)
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


def monte_carlo(chunk, sampler, predictor, n_repeats, n_steps=12):
    res = torch.zeros(chunk.shape[0], sampler.k, dtype=torch.float32, device=predictor.device)
    for _ in tqdm(range(n_repeats)):
        h = None
        x = chunk.clone()
        for k in range(n_steps):
            x, r = sampler(x)
            x, h = predictor(x, h)
            res += r
    return res / n_repeats


@hydra.main(version_base=None)
def main(conf: DictConfig):
    # hydra.initialize(version_base=None, config_path="conf")
    # conf = hydra.compose("pt_inference", overrides=["device=cpu"])

    predictor = OneStepPredictor(conf)
    logger.info(f"Model weights restored from: {predictor.ckpt}.")
    logger.info(repr(predictor.seq_encoder))
    logger.info(repr(predictor.head))

    dataset = ChunkedDataset(conf)
    logger.info(f"Initial dataset {dataset.data.shape} splitted into {dataset.n_chunks} chunks.")

    sampler = Sampler(conf)
    res = []
    for i in range(dataset.n_chunks):
        res.append(monte_carlo(dataset[i], sampler, predictor, conf.mc.repeats, conf.mc.steps))
        logger.info(f"Done chunk {i + 1} from {dataset.n_chunks}.")

    res = torch.cat(res, dim=0).cpu().numpy()
    df = {"id": dataset.ids, "target_mc": res.dot(dataset.cat_weight)}
    df.update({f"n_{i}": res[:, i] for i in range(res.shape[1])})
    pd.DataFrame(df).merge(dataset.target, on="id").to_csv("monte_carlo.csv", header=True, index=False)


if __name__ == "__main__":
    main()
