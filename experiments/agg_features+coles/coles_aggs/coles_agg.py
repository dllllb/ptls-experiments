import torch

from ptls.frames.abs_module import ABSModule
from ptls.frames.coles.losses.contrastive_loss import ContrastiveLoss
from ptls.frames.coles.metric import BatchRecallTopK
from ptls.frames.coles.sampling_strategies.hard_negative_pair_selector import HardNegativePairSelector
from ptls.nn.head import Head
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.frames.coles_aggs import ABSWithAggFeatsModule
from functools import partial
from ptls.frames.abs_module import ABSModule
from ptls.frames.abs_module import ABSModule
from ptls.data_load.padded_batch import PaddedBatch
from functools import partial


class CoLESWithAggModule(ABSModule):
    def __init__(self,
                 seq_encoder: SeqEncoderContainer = None,
                 agg_encoder=None,
                 agg_loss=None,
                 agg_coeff=0.5,
                 coles_coeff=0.5,
                 head=None,
                 loss=None,
                 validation_metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None):
      
        if head is None:
            head = Head(use_norm_encoder=True)

        if loss is None:
            loss = ContrastiveLoss(margin=0.5,
                                   sampling_strategy=HardNegativePairSelector(neg_count=5))

        if validation_metric is None:
            validation_metric = BatchRecallTopK(K=4, metric='cosine')

        super().__init__(validation_metric,
                         seq_encoder,
                         loss,
                         optimizer_partial,
                         lr_scheduler_partial
                         )

        self._head = head
        self._loss = loss
        self._agg_loss = agg_loss
        self._seq_encoder = seq_encoder
        self._agg_encoder = agg_encoder
        self._seq_encoder.is_reduce_sequence = self.is_requires_reduced_sequence
        self._validation_metric = validation_metric

        self._optimizer_partial = optimizer_partial
        self._lr_scheduler_partial = lr_scheduler_partial

        self._agg_coeff = agg_coeff
        self._coles_coeff = coles_coeff

    @property
    def metric_name(self):
        return 'recall_top_k'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y

    # def forward(self, x):
    #     coles_out = self._seq_encoder(x)
    #     agg_out = self._agg_encoder(x)
    #     out = torch.cat((coles_out, agg_out), dim=1)
    #     return out

    def training_step(self, batch, _):
        coles_in, agg_in = batch[0], batch[1]
        y_h, y = self.shared_step(*coles_in)
        agg_out = self._agg_encoder(agg_in)

        coles_loss = self._loss(y_h, y)
        agg_loss = self._agg_loss(y_h, agg_out)

        loss = self._coles_coeff*coles_loss + self._agg_coeff*agg_loss

        self.log('coles_loss', coles_loss)
        self.log('agg_loss', agg_loss)

        if type(coles_in) is tuple:
            x, y = coles_in
            if isinstance(x, PaddedBatch):
                self.log('seq_len', x.seq_lens.float().mean(), prog_bar=True)
        else:
            # this code should not be reached
            self.log('seq_len', -1, prog_bar=True)
            raise AssertionError('batch is not a tuple')
        return loss

    def validation_step(self, batch, _):
        coles_in, agg_in = batch[0], batch[1]
        y_h, y = self.shared_step(*coles_in)
        self._validation_metric(y_h, y)