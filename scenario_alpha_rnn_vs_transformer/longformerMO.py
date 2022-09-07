# def cpc_collate_fn(batch):
#     xs, ts = [], []
#     for rec in batch:
#         x, t = {k: v[:-2] for k, v in rec.items()}, {k: v[-2:] for k, v in rec.items()}
#         xs.append(x)
#         ts.append(t)
#     return padded_collate_wo_target(xs), padded_collate_wo_target(ts)


import numpy as np
import torch
import pytorch_lightning as pl
 
from omegaconf import DictConfig
from transformers import LongformerConfig, LongformerModel
 
from ptls.data_load.padded_batch import PaddedBatch
from torchmetrics import MeanMetric
from ptls.frames.bert.losses.query_soft_max import QuerySoftmaxLoss
from torch.nn import BCELoss
 

class ContrastivePredictionHead(torch.nn.Module):
   
    def __init__(self, embeds_dim, drop_p=0.1):
       
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(embeds_dim, embeds_dim, bias=True)
        )
 
    def forward(self, x):
        return self.head(x)


class MLMCPCPretrainModule(pl.LightningModule):
    def __init__(self,
                 trx_encoder: torch.nn.Module,
                 hidden_size: int,
                 loss_temperature: float,
                 total_steps: int,
                 max_lr: float = 0.001,
                 weight_decay: float = 0.0,
                 pct_start: float = 0.1,
                 norm_predict: bool = False,
                 num_attention_heads: int = 16,
                 intermediate_size: int = 128,
                 num_hidden_layers: int = 2,
                 attention_window: int = 16,
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings: int = 4000,
                 replace_proba: float = 0.1,
                 neg_count: int = 1,
                 log_logits: bool = False,
                 weight_mlm: float = 0.5,
                 weight_cpc: float = 0.5,
                 encode_seq = False
                 ):
        """
 
        Parameters
        ----------
        trx_encoder:
            Module for transform dict with feature sequences to sequence of transaction representations
        hidden_size:
            Output size of `trx_encoder`. Hidden size of internal transformer representation
        loss_temperature:
             temperature parameter of `QuerySoftmaxLoss`
        total_steps:
            total_steps expected in OneCycle lr scheduler
        max_lr:
            max_lr of OneCycle lr scheduler
        weight_decay:
            weight_decay of Adam optimizer
        pct_start:
            % of total_steps when lr increase
        norm_predict:
            use l2 norm for transformer output or not
        num_attention_heads:
            parameter for Longformer
        intermediate_size:
            parameter for Longformer
        num_hidden_layers:
            parameter for Longformer
        attention_window:
            parameter for Longformer
        max_position_embeddings:
            parameter for Longformer
        replace_proba:
            probability of masking transaction embedding
        neg_count:
            negative count for `QuerySoftmaxLoss`
        log_logits:
            if true than logits histogram will be logged. May be useful for `loss_temperature` tuning
        encode_seq:
            if true then outputs zero element of transformer i.e. encodes whole sequence. Else returns all outputs of transformer except 0th.
        """
 
        super().__init__()
        self.save_hyperparameters(logger=False)
 
        self.trx_encoder = trx_encoder
 
        self.token_cls = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
        self.token_mask = torch.nn.Parameter(torch.randn(1, 1, hidden_size), requires_grad=True)
 
        self.transf = LongformerModel(
            config=LongformerConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=4,
                max_position_embeddings=self.hparams.max_position_embeddings,
                attention_window=attention_window,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob
            ),
            add_pooling_layer=False,
        )
       
        self.cpc_head1 = ContrastivePredictionHead(embeds_dim=hidden_size)
        self.cpc_head2 = ContrastivePredictionHead(embeds_dim=hidden_size)
       
        self.mlm_loss = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)
        self.cpc_loss = QuerySoftmaxLoss(temperature=loss_temperature, reduce=False)
       
        self.weight_mlm = weight_mlm
        self.weight_cpc = weight_cpc
 
        self.train_mlm_loss = MeanMetric(compute_on_step=False)
        self.valid_mlm_loss = MeanMetric(compute_on_step=False)
        
        self.train_cpc_loss = MeanMetric(compute_on_step=False)
        self.valid_cpc_loss = MeanMetric(compute_on_step=False)
 
        
        self.encode_seq = encode_seq
 
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams.max_lr,
                                 weight_decay=self.hparams.weight_decay,
                                 )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.hparams.max_lr,
            total_steps=self.hparams.total_steps,
            pct_start=self.hparams.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step'}
        return [optim], [scheduler]
 
    def get_mask(self, attention_mask):
        return torch.bernoulli(attention_mask.float() * self.hparams.replace_proba).bool()
 
    def mask_x(self, x, attention_mask, mask):
        shuffled_tokens = x[attention_mask.bool()]
        B, T, H = x.size()
        ix = torch.multinomial(torch.ones(shuffled_tokens.size(0)), B * T, replacement=True)
        shuffled_tokens = shuffled_tokens[ix].view(B, T, H)
 
        rand = torch.rand(B, T, device=x.device).unsqueeze(2).expand(B, T, H)
        replace_to = torch.where(
            rand < 0.8,
            self.token_mask.expand_as(x),  # [MASK] token 80%
            torch.where(
                rand < 0.9,
                shuffled_tokens,  # random token 90%
                x,  # unchanged 10%
            )
        )
        return torch.where(mask.bool().unsqueeze(2).expand_as(x), replace_to, x)
 
    def forward(self, z: PaddedBatch):
        z = self.trx_encoder(z)
        
        B, T, H = z.payload.size()
        device = z.payload.device
 
        if self.training:
            start_pos = np.random.randint(0, self.hparams.max_position_embeddings - T - 1, 1)[0]
        else:
            start_pos = 0
 
        inputs_embeds = z.payload
        attention_mask = z.seq_len_mask.float()
 
        inputs_embeds = torch.cat([
            self.token_cls.expand(inputs_embeds.size(0), 1, H),
            inputs_embeds,
        ], dim=1)
        attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            attention_mask,
        ], dim=1)
        position_ids = torch.arange(T + 1, device=z.device).view(1, -1).expand(B, T + 1) + start_pos
        global_attention_mask = torch.cat([
            torch.ones(inputs_embeds.size(0), 1, device=device),
            torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1) - 1, device=device),
        ], dim=1)
 
        out = self.transf(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state
 
        if self.hparams.norm_predict:
            out = out / (out.pow(2).sum(dim=-1, keepdim=True) + 1e-9).pow(0.5)
       
        # CPC Predictions
        cpc_preds1 = self.cpc_head1.forward(out[:, 0])
        cpc_preds2 = self.cpc_head2.forward(out[:, 0])
       
        if self.encode_seq:
            return out[:, 0]
        else:
            return PaddedBatch(out[:, 1:], z.seq_lens), [cpc_preds1, cpc_preds2]
 
    def get_neg_ix(self, mask):
        """Sample from predicts, where `mask == True`, without self element.
        sample from predicted tokens from batch
        """
        mask_num = mask.int().sum()
        mn = 1 - torch.eye(mask_num, device=mask.device)
        neg_ix = torch.multinomial(mn, max(1, int(mn.shape[-1]*0.09)))  # self.hparams.neg_count
 
        b_ix = torch.arange(mask.size(0), device=mask.device).view(-1, 1).expand_as(mask)[mask][neg_ix]
        t_ix = torch.arange(mask.size(1), device=mask.device).view(1, -1).expand_as(mask)[mask][neg_ix]
        return b_ix, t_ix
 
    def loss_mlm_cpc(self, x: PaddedBatch, y: PaddedBatch, is_train_step):
       
        mask = self.get_mask(x.seq_len_mask)
        masked_x = self.mask_x(x.payload, x.seq_len_mask, mask)
        out, preds  = self.forward(PaddedBatch(masked_x, x.seq_lens))
       
        # MlM Part
        out = out.payload
        mask = mask
        target = x.payload[mask].unsqueeze(1)  # N, 1, H
        predict = out[mask].unsqueeze(1)  # N, 1, H
        neg_ix = self.get_neg_ix(mask)
        negative = out[neg_ix[0], neg_ix[1]]  # N, nneg, H
        loss_mlm = self.mlm_loss(target, predict, negative)
 
        if is_train_step and self.hparams.log_logits:
            with torch.no_grad():
                logits = self.mlm_loss.get_logits(target, predict, negative)
            self.logger.experiment.add_histogram('mlm/logits',
                                                 logits.flatten().detach(), self.global_step)
           
            
        # CPC Part 
        targets, predicts, negatives = [], [], []
        for i in range(2):
            target = y.payload[:, i].unsqueeze(1)  # B, 1, H
            predict = preds[i].unsqueeze(1)  # B, 1, H
 
            # Sample negatives along batch_size dimension
            batch_size = predict.size(0)
            mn = 1 - torch.eye(batch_size, device=target.device)
            neg_ix = torch.multinomial(mn, max(1, int(mn.shape[-1]*0.07)))  # self.hparams.neg_count
            negative = preds[i][neg_ix, :]  # B, nneg, H
            targets.append(target)
            predicts.append(predict)
            negatives.append(negative)
       
        targets = torch.concat(targets, dim=0)
        predicts = torch.concat(predicts, dim=0)
        negatives = torch.concat(negatives, dim=0)
       
        # Feed contrastive loss with negatives
        loss_cpc = self.cpc_loss(targets, predicts, negatives) 
    
        return loss_mlm, loss_cpc
 
    def training_step(self, batch, batch_idx):
        x_trx, y_trx = batch
       
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        y_trx = self.trx_encoder(y_trx)  #     B, 3, H
        loss_mlm, loss_cpc = self.loss_mlm_cpc(z_trx, y_trx, is_train_step=True)
        self.train_mlm_loss(loss_mlm)
        self.train_cpc_loss(loss_cpc)
        loss_mlm = loss_mlm.mean()
        loss_cpc = loss_cpc.mean()
        self.log(f'mlm/loss', loss_mlm)
        self.log(f'cpc/loss', loss_cpc)
        loss = self.weight_cpc*loss_cpc + self.weight_mlm*loss_mlm
        return loss
 
    def validation_step(self, batch, batch_idx):
        x_trx, y_trx = batch
        z_trx = self.trx_encoder(x_trx)  # PB: B, T, H
        y_trx = self.trx_encoder(y_trx)
        loss_mlm, loss_cpc = self.loss_mlm_cpc(z_trx, y_trx, is_train_step=False)
        self.valid_cpc_loss(loss_cpc)
        self.valid_mlm_loss(loss_mlm)
 
    def training_epoch_end(self, _):
        self.log(f'mlm/train_mlm_loss', self.train_mlm_loss, prog_bar=False)
        self.log(f'cpc/train_cpc_loss', self.train_cpc_loss, prog_bar=False)
 
    def validation_epoch_end(self, _):
        self.log(f'mlm/valid_mlm_loss', self.valid_mlm_loss, prog_bar=True)
        self.log(f'cpc/valid_cpc_loss', self.valid_cpc_loss, prog_bar=False)
