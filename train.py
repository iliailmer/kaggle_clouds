import numpy as np
from catalyst.dl.callbacks import CriterionAggregatorCallback
from catalyst.contrib.criterion import DiceLoss, IoULoss
import torch
from catalyst.contrib.optimizers import RAdam, Lookahead
from utils import (
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation
)
import argparse
import pandas as pd
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import utils
from catalyst.dl.callbacks import (EarlyStoppingCallback,
                                   CriterionCallback,
                                   OptimizerCallback,
                                   IouCallback,
                                   DiceCallback,
                                   CheckpointCallback)
# import torch
from torch import optim
from torch.utils.data import DataLoader
from dataloader import CloudDataset

import segmentation_models_pytorch as smp
from catalyst.utils.seed import set_global_seed
from catalyst.utils.torch import prepare_cudnn

set_global_seed(2019)
prepare_cudnn(deterministic=True)

parser = argparse.ArgumentParser("PyTorch Segmentation Pipeline")
args = parser.add_argument('-E', '--epochs', default=1, type=int)
args = parser.add_argument('-F', '--fold', default=1, type=int)
args = parser.add_argument('-C', '--checkpoint', default=False, type=bool)
args = parser.add_argument('-M', '--model', default='AlbuNet', type=str)
args = parser.add_argument('-A', '--encoder', default='resnet18', type=str)
args = parser.add_argument('-P', '--pretrained', default=True, type=bool)
args = parser.add_argument('--lr', default=1e-4, type=float)
args = parser.add_argument('--lr_e', default=1e-4, type=float)
args = parser.add_argument('--lr_d', default=1e-4, type=float)
args = parser.add_argument('--bs', default=4, type=int)
args = parser.add_argument('--dice-weight', default=0.5, type=float)
args = parser.parse_args()
path = '../data/cloud_data'

models = {'smpunet': (smp.Unet, {'encoder_name': args.encoder,
                                 'encoder_weights': 'imagenet',
                                 'classes': 4,
                                 'activation': None}),
          'linknet': (smp.Linknet, {'encoder_name': args.encoder,
                                    'encoder_weights': 'imagenet',
                                    'classes': 4,
                                    'activation': None}),
          'fpn': (smp.FPN, {'encoder_name': args.encoder,
                            'encoder_weights': 'imagenet',
                            'classes': 4,
                            'activation': None})}

preprocessing_fn = smp.encoders.get_preprocessing_fn(
    encoder_name=args.encoder, pretrained='imagenet')

model = models[args.model.lower()][0](**models[args.model.lower()][1]).cuda()

layerwise_params = {
    "encoder*": dict(lr=args.lr_e, weight_decay=0.00003)}
model_params = utils.process_model_params(
    model, layerwise_params=layerwise_params)

base_optimizer = RAdam(model_params, lr=args.lr_d, weight_decay=1e-6)
optimizer = Lookahead(base_optimizer)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5)
criteria = {'dice': DiceLoss(),
            'iou': IoULoss(),
            'bce': torch.nn.BCEWithLogitsLoss()}

train = pd.read_csv(f'train_preprocessed.csv')
train_ids = pd.read_csv(f'./folds/fold_{args.fold}_train.csv').values.ravel()
valid_ids = pd.read_csv(f'./folds/fold_{args.fold}_val.csv').values.ravel()
num_workers = 2
bs = args.bs
train_dataset = CloudDataset(df=train,
                             path=path,
                             datatype='train',
                             preload=False,
                             img_ids=train_ids,
                             transforms=get_training_augmentation(
                                 p=max(0.5, np.random.beta(1, 2))),
                             preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = CloudDataset(df=train,
                             path=path,
                             datatype='valid',
                             preload=False,
                             img_ids=valid_ids,
                             transforms=get_validation_augmentation(),
                             preprocessing=get_preprocessing(preprocessing_fn))

train_loader = DataLoader(train_dataset, batch_size=bs,
                          shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs,
                          shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

runner = SupervisedRunner(model=model,
                          device='cuda',
                          input_key='image',
                          input_target_key='mask')
logdir = f'./logs/{args.model}/fold_{args.fold}'
num_epochs = args.epochs
callbacks = [CriterionCallback(input_key='mask',
                               multiplier=1.,
                               prefix='loss_dice',
                               criterion_key='dice'),
             CriterionCallback(input_key='mask',
                               multiplier=0.8,
                               prefix='loss_iou',
                               criterion_key='iou'),
             CriterionCallback(input_key='mask',
                               prefix='loss_bce',
                               multiplier=0.8,
                               criterion_key='bce'),
             CriterionAggregatorCallback(prefix='loss',
                                         loss_keys=["loss_dice",
                                                    "loss_iou", "loss_bce"],
                                         loss_aggregate_fn="sum"),
             DiceCallback(input_key='mask'),
             IouCallback(input_key='mask'),
             OptimizerCallback(accumulation_steps=64),
             EarlyStoppingCallback(patience=10, min_delta=0.001),
             ]
if args.checkpoint:
    callbacks.append(CheckpointCallback(
        resume=f'{logdir}/checkpoints/best_full.pth'))
runner.train(
    model=model,
    criterion=criteria,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    main_metric='dice',
    minimize_metric=False,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)
