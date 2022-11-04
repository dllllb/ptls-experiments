import logging
from copy import deepcopy

import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import DictConfig

from ptls.data_load.padded_batch import PaddedBatch
logger = logging.getLogger(__name__)

class MapDataset(torch.utils.data.Dataset):
    def __init__(self, iter_parent):
        super().__init__()
        self.parent=list(tqdm(iter(iter_parent)))
        np.random.shuffle(self.parent)
    def __len__(self):
        return len(self.parent)
    def __getitem__(self, idx):
        return self.parent[idx]

class SequenceToTarget(pl.LightningModule):
    def __init__(self,
                 seq_encoder,
                 head: torch.nn.Module=None,
                 loss: torch.nn.Module=None,
                 dropout=0.1,
                 metric_list: torchmetrics.Metric=None,
                 optimizer_partial=None,
                 lr_scheduler_partial=None,
                 pretrained_lr=None,
                 train_update_n_steps=None,
                 ):

        super().__init__()

        self.save_hyperparameters(ignore=[
            'seq_encoder', 'head', 'loss', 'metric_list', 'optimizer_partial', 'lr_scheduler_partial'])
        
        self.seq_encoder = seq_encoder
        self.head = head
        self.loss = loss
        self.dropout = torch.nn.Dropout(dropout)
        if type(metric_list) is dict or type(metric_list) is DictConfig:
            metric_list = [(k, v) for k, v in metric_list.items()]
        else:
            if type(metric_list) is not list:
                metric_list = [metric_list]
            metric_list = [(m.__class__.__name__, m) for m in metric_list]

        self.train_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])
        self.valid_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])
        self.test_metrics = torch.nn.ModuleDict([(name, deepcopy(mc)) for name, mc in metric_list])
        
        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial

    def forward(self, x):
        x = self.seq_encoder.trx_encoder(x)
        x = self.seq_encoder.forward(x).payload
        x = self.dropout(x)

        x = self.head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        loss = self.loss(y_h, y)
        self.log(f'loss/train', loss)
        for name, mf in self.train_metrics.items():
            mf(y_h, y)
        return loss

    def training_epoch_end(self, outputs):
        for name, mf in self.train_metrics.items():
            self.log(f'{name}/train', mf.compute(), prog_bar=False)
        for name, mf in self.train_metrics.items():
            mf.reset()

    def validation_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        self.log('loss/val', self.loss(y_h, y))
        for name, mf in self.valid_metrics.items():
            mf(y_h, y)

    def validation_epoch_end(self, outputs):
        for name, mf in self.valid_metrics.items():
            self.log(f'{name}/val', mf.compute(), prog_bar=False)
        for name, mf in self.valid_metrics.items():
            mf.reset()

    def test_step(self, batch, _):
        x, y = batch
        y_h = self(x)
        for name, mf in self.test_metrics.items():
            mf(y_h, y)

    def test_epoch_end(self, outputs):
        for name, mf in self.test_metrics.items():
            value = mf.compute().item()
            self.log(f'{name}/test', value, prog_bar=False)
        for name, mf in self.test_metrics.items():
            mf.reset()
        
    def configure_optimizers(self):
        if self.hparams.pretrained_lr is not None:
            if self.hparams.pretrained_lr == 'freeze':
                for p in self.seq_encoder.parameters():
                    p.requires_grad = False
                logger.info('Created optimizer with frozen encoder')
                parameters = self.parameters()
            else:
                parameters = [
                    {'params': self.seq_encoder.parameters(), 'lr': self.hparams.pretrained_lr},
                    {'params': self.head.parameters()},  # use predefined lr from `self.optimizer_partial`
                ]
                logger.info('Created optimizer with two lr groups')
        else:
            parameters = self.parameters()

        optimizer = self.optimizer_partial(parameters)
        scheduler = self.lr_scheduler_partial(optimizer)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

if __name__ == '__main__':
    pass
