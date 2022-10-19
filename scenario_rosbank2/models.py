import logging
from itertools import chain
import numpy as np

import torch
from ptls.data_load import PaddedBatch
from ptls.frames.supervised import SequenceToTarget
from ptls.nn.trx_encoder.batch_norm import RBatchNormWithLens, RBatchNorm

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


class GlobalPoolingHead(torch.nn.Module):
    def __init__(self,
                 use_mean: bool= True,
                 use_std: bool= True,
                 use_min: bool= True,
                 use_max: bool= True,
                 ):
        super().__init__()

        self.use_mean = use_mean
        self.use_std = use_std
        self.use_min = use_min
        self.use_max = use_max

        self.eps = 1e-9

    def forward(self, x: PaddedBatch):
        """
        input: B, T, H
        output: B, H * n
            where n is count of active `self.use_` flags
        """
        processed = []
        seq_lens = x.seq_lens.unsqueeze(1)
        if self.use_mean:
            mean_ = x.payload.sum(dim=1).div(seq_lens + self.eps)
            processed.append(mean_)
        if self.use_std:
            a = x.payload.pow(2).sum(dim=1) - x.payload.sum(dim=1).pow(2).div(seq_lens + self.eps)
            a = torch.clamp(a, min=0.0)
            std_ = a.div(torch.clamp(seq_lens - 1, min=0.0) + self.eps).pow(0.5)  # std
            processed.append(std_)
        if self.use_max:
            max_ = x.payload.masked_fill(~x.seq_len_mask.bool().unsqueeze(2), np.float32('-inf')).max(dim=1).values
            processed.append(max_)
        if self.use_min:
            min_ = x.payload.masked_fill(~x.seq_len_mask.bool().unsqueeze(2), np.float32('inf')).min(dim=1).values
            processed.append(min_)
        return torch.cat(processed, dim=1)
