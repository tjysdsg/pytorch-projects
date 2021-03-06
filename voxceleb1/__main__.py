import os
import glob
import argparse
import collections
import torch
from voxceleb1 import dataset
from voxceleb1 import dataloader
from voxceleb1.dataloader.subset import random_split
import voxceleb1.module.metric as module_metric
import voxceleb1.module.model as module_model
import voxceleb1.module.loss as module_loss
from voxceleb1.utils import setup_seed
from voxceleb1.config import MODEL_DIR
from voxceleb1.utils.parse_config import ConfigParser
from voxceleb1.trainer import Trainer


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    if config['seed'] is not None:
        setup_seed(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load and split data into trainset and validset by valid_split
    fullset = config.init_obj('dataset', dataset)
    trainset, validset = random_split(fullset, config['dataset']['valid_split'])
    trainloader = config.init_obj('dataloader', dataloader, trainset)
    if validset is None:
        validloader = None
    else:
        validloader = config.init_obj('dataloader', dataloader, validset)

    # build model architecture, then print to console
    model = config.init_obj('model', module_model)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # delete all under "lr_scheduler": {} in config.json to disable scheduler
    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, config=config, trainloader=trainloader,
                      validloader=validloader, lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Voxceleb1 speaker identification')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-a', '--auto-resume', default=True, action='store_true',
                      help='find the latest checkpoint and auto resume, will override --resume')
    parsed_args = args.parse_args()

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-n', '--name'], type=str, target='name'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='dataloader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    config['trainer']['save_dir'] = MODEL_DIR

    # auto resume
    if parsed_args.auto_resume:
        print("Auto resume is enabled")
        checkpoint_dir = os.path.join(MODEL_DIR, config['name'], 'chkpt')
        checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))

        if len(checkpoints) > 0:
            # if there is a best model, use it to train
            if 'model_best.pth' in checkpoints:
                config['resume'] = os.path.join(checkpoint_dir, 'model_best.pth')
            else:
                checkpoint_epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]), checkpoints))
                ch = 'chkpt_{:03d}.pth'.format(max(checkpoint_epochs))
                config['resume'] = os.path.join(checkpoint_dir, ch)
    else:
        config['resume'] = None
    main(config)
