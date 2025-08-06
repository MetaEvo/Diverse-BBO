import torch
from trainer import Trainer
from tester import *
from config import get_config
from logger import *
import shutil
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    config = get_config()
    assert ((config.train is not None) +
            (config.rollout is not None) +
            (config.test is not None) +
            (config.run_experiment is not None) +
            (config.mgd_test is not None) +
            (config.mte_test is not None)) == 1, \
        'Among train, rollout, test, run_experiment, mgd_test & mte_test, only one mode can be given at one time.'

    # train
    if config.train:
        torch.set_grad_enabled(True)
        trainer = Trainer(config)
        trainer.train_new()

    # test
    if config.test:
        torch.set_grad_enabled(False)
        tester = Tester(config)
        tester.test_1()
        post_processing_test_statics(config.test_log_dir, Logger(config))
