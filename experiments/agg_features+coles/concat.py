import os
import pickle

import numpy as np
import pandas as pd

coles_embs = pd.read_pickle("/home/vorobev/ptls-experiments/scenario_age_pred/data/coles_agg.pickle")
agg_embs = pd.read_pickle("/home/vorobev/ptls-experiments/scenario_age_pred/data/agg_feat_embed.pickle")

agg_embs = agg_embs.drop(columns=['cl_id'])
embs = pd.concat([coles_embs, agg_embs], axis=1)

embs.to_pickle('/home/vorobev/ptls-experiments/scenario_age_pred/data/coles_vicreg_agg_concat.pickle')

coles_embs = pd.read_pickle("/home/vorobev/ptls-experiments/scenario_age_pred/data/coles.pickle")
agg_embs = pd.read_pickle("/home/vorobev/ptls-experiments/scenario_age_pred/data/agg_feat_embed.pickle")

agg_embs = agg_embs.drop(columns=['cl_id'])
embs = pd.concat([coles_embs, agg_embs], axis=1)

embs.to_pickle('/home/vorobev/ptls-experiments/scenario_age_pred/data/coles_agg_concat.pickle')
# print(embs.shape)
# print(embs.columns)
# print(len(embs))