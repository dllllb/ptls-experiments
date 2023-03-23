from functools import reduce
from operator import iadd
import torch

from ptls.frames.coles import ColesDataset
from ptls.data_load.utils import collate_feature_dict
from ptls.frames.coles.split_strategy import AbsSplit

class ColesWithAggFeatsDataset(ColesDataset):
  def __init__(self,
                 data,
                 splitter: AbsSplit,
                 col_time='event_time',
                 *args, **kwargs):
        super().__init__(
            data,
            splitter,
            *args, **kwargs)  # required for mixin class

        self.data = data
        self.splitter = splitter
        self.col_time = col_time

  def __getitem__(self, idx):
        feature_arrays = self.data[idx]
        return self.get_splits(feature_arrays), feature_arrays

  def __iter__(self):
        for feature_arrays in self.data:
            yield self.get_splits(feature_arrays), feature_arrays 

  @staticmethod
  def collate_fn(batch):
        coles_in, agg_in = [x[0] for x in batch], [x[1] for x in batch]
        class_labels = [i for i, class_samples in enumerate(coles_in) for _ in class_samples]
        coles_in = reduce(iadd, coles_in)
        padded_batch = collate_feature_dict(coles_in)
        padded_batch_agg = collate_feature_dict(agg_in)
 
        return (padded_batch, torch.LongTensor(class_labels)), (padded_batch_agg)