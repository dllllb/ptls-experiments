import os

import luigi
import datetime
import hydra

from omegaconf import DictConfig, OmegaConf
from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from embeddings_validation.tasks.fold_splitter import FoldSplitter


@hydra.main()
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)
    orig_cwd = hydra.utils.get_original_cwd()

    conf.workers = conf.get('workers')
    if conf.workers is None: raise AttributeError
    conf.total_cpu_count = conf.get('total_cpu_count')
    if conf.total_cpu_count is None: raise AttributeError

    conf.split_only = conf.get('split_only', False)
    conf.local_scheduler = conf.get('local_sheduler', True)
    conf.log_level = conf.get('log_level', 'INFO')

    conf = Config.get_conf(conf, orig_cwd + '/' + conf.conf_path)

    if conf['split_only']:
        task = FoldSplitter(
            conf=conf,
        )
    else:
        task = ReportCollect(
            conf=conf,
            total_cpu_count=conf['total_cpu_count'],
        )
    luigi.build([task], workers=conf['workers'],
                        local_scheduler=conf['local_scheduler'],
                        log_level=conf['log_level'])


if __name__ == '__main__':
    main()

