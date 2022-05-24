import os

import logging
from glob import glob
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Config:
    VALID_TRAIN_TEST = 'train-test'
    VALID_CROSS_VAL = 'crossval'

    ON_ERROR_FAIL = 'fail'
    ON_ERROR_SKIP = 'skip'

    def __init__(self, conf=None, root_path=None):
        self.conf = conf
        self.root_path = root_path

    @classmethod
    def get_conf(cls, conf: DictConfig, abs_conf_path: str):
        logger.info('Load config from "{0}"'.format(conf['conf_path']))
        root_path = os.path.dirname(abs_conf_path)
        return cls(conf=conf, root_path=root_path)

    def __getitem__(self, item):
        return self.conf[item]

    def _read_enabled(self, key):
        return {name: {k: v for k, v in params.items() if k != 'enabled'}
                for name, params in self.conf[key].items() if params['enabled']}

    @property
    def work_dir(self):
        return os.path.join(self.root_path, self.conf['environment']['work_dir'])

    def resolve_path_wc(self, path_wc):
        for path in glob(os.path.join(self.root_path, path_wc)):
            file_name = os.path.basename(path)
            file_name = os.path.splitext(file_name)[0]
            yield file_name, path

    @property
    def features(self):
        features = self._read_enabled('features')
        if 'auto_features' in self.conf:
            auto_features = {name: {'read_params': {'file_name': path}, 'target_options': {}}
                             for path_wc in self.conf['auto_features']
                             for name, path in self.resolve_path_wc(path_wc)}
            features.update(auto_features)
        return features

    @property
    def external_scores(self):
        ext_scores = {k: v for k, v in self.conf['external_scores'].items()}
        if 'auto_scores' in self.conf:
            auto_scores = {name: path
                           for path_wc in self.conf['auto_scores']
                           for name, path in self.resolve_path_wc(path_wc)}
            ext_scores.update(auto_scores)

        return ext_scores

    @property
    def models(self):
        return self._read_enabled('models')

    @property
    def validation_schema(self):
        split_params = self.conf['split']

        if 'train_id' not in split_params:
            raise AttributeError(f'There is no "train" key in "split" config')

        if 'valid_id' in split_params:
            if 'n_iteration' not in split_params:
                raise AttributeError(f'"n_iteration" should be defined '
                                     f'when both "train" and "valid" keys are presented in "split" config')
            return self.VALID_TRAIN_TEST
        else:
            for attr in ['cv_split_count', 'is_stratify', 'random_state']:
                if attr not in split_params:
                    raise AttributeError(f'"{attr}" should be defined when only "train" key is presented '
                                         f'and "valid" key is absent in "split" config')
            return self.VALID_CROSS_VAL

    @property
    def folds(self):
        split_params = self.conf['split']
        validation_schema = self.validation_schema

        if validation_schema == self.VALID_TRAIN_TEST:
            return [i for i in range(split_params['n_iteration'])]
        if validation_schema == self.VALID_CROSS_VAL:
            return [i for i in range(split_params['cv_split_count'])]

        raise AssertionError(f'Unknown validation_schema: {validation_schema}')

    @property
    def metrics(self):
        return self._read_enabled('metrics')

    @property
    def error_handling(self):
        error_handling = self.conf['report']['error_handling']
        if error_handling == self.ON_ERROR_FAIL:
            return self.ON_ERROR_FAIL
        if error_handling == self.ON_ERROR_SKIP:
            return self.ON_ERROR_SKIP

        raise AttributeError(f'Unknown error_handling: "{error_handling}"')
