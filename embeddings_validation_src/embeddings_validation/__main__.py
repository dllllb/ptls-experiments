import os

import luigi
import argparse
import datetime
import hydra

from collections import namedtuple
from omegaconf import DictConfig, OmegaConf
from embeddings_validation import ReportCollect
from embeddings_validation.config import Config
from embeddings_validation.tasks.fold_splitter import FoldSplitter


def read_extra_conf(file_name, conf_extra):
    conf = Config.read_file(file_name, conf_extra)
    name, ext = os.path.splitext(file_name)
    tmp_file_name = f'{name}_{datetime.datetime.now().timestamp():.0f}{ext}'
    conf.save_tmp_copy(tmp_file_name)
    return tmp_file_name


@hydra.main()
def main(conf: DictConfig):
    OmegaConf.set_struct(conf, False)
    orig_cwd = hydra.utils.get_original_cwd()

    conf.workers = conf.get('workers', '8 ???')
    conf.total_cpu_count = conf.get('total_cpu_count', '8 ???')
    conf.split_only = conf.get('split_only', True)
    conf.local_scheduler = conf.get('local_sheduler', True)
    conf.log_level = conf.get('log_level', 'INFO')

    conf = Config.get_conf(conf, orig_cwd + '/' + conf.conf_path)

    if conf.conf.split_only:
        task = FoldSplitter(
            conf=conf,
        )
    else:
        task = ReportCollect(
            conf=conf,
            total_cpu_count=conf.conf.total_cpu_count,
        )
    luigi.build([task], workers=conf.conf.workers, local_scheduler=conf.conf.local_scheduler, log_level=conf.conf.log_level)


if __name__ == '__main__':
    main()

