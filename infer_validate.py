from models.linknet_attention import LinkNetGated
import argparse
from dataloader.loader import CloudDataset
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner

from catalyst.dl.callbacks import (InferCallback,
                                   CheckpointCallback)
import tqdm
import cv2
import pandas as pd
import numpy as np
from utils.utils import (get_preprocessing, post_process,
                         get_validation_augmentation, dice)
import gc
import json
import segmentation_models_pytorch as smp

parser = argparse.ArgumentParser("PyTorch Segmentation Pipeline")
args = parser.add_argument('-F', '--fold', default=1, type=int)
args = parser.add_argument('-M', '--model', default='AlbuNet', type=str)
args = parser.add_argument('-A', '--encoder', default='resnet18', type=str)
args = parser.add_argument('--bs', default=4, type=int)
args = parser.parse_args()
path = '../data/cloud_data'

train = pd.read_csv(f'train_preprocessed.csv')
valid_ids = pd.read_csv(f'./folds/fold_{args.fold}_val.csv').values.ravel()

num_workers = 2
bs = args.bs
preprocessing_fn = smp.encoders.get_preprocessing_fn(
    encoder_name=args.encoder, pretrained='imagenet')
valid_dataset = CloudDataset(df=train,
                             path=path,
                             datatype='valid',
                             preload=False,
                             img_ids=valid_ids,
                             filter_bad_images=True,
                             transforms=get_validation_augmentation(),
                             preprocessing=get_preprocessing(preprocessing_fn))
valid_loader = DataLoader(valid_dataset,
                          batch_size=bs,
                          shuffle=False,
                          num_workers=num_workers)

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
                            'activation': None}),
          'attn_linknet': (LinkNetGated, {'num_classes': 4,
                                          'in_channels': 3
                                          })}
model = models[args.model.lower()][0](**models[args.model.lower()][1])
encoded_pixels = []
loaders = {"infer": valid_loader}
logdir = f'./logs/{args.model}/fold_{args.fold}'
gc.collect()
runner = SupervisedRunner(model=model,
                          device='cuda',
                          input_key='image',
                          input_target_key='mask')
runner.infer(
    model=model,
    loaders=loaders,
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"),
        InferCallback()
    ],
    # fp16={"opt_level": "O1"},
)
valid_masks = []
probabilities = np.zeros((2220, 350, 525))
for i, (batch, output) in enumerate(tqdm.tqdm(zip(
        valid_dataset, runner.callbacks[0].predictions["logits"]))):
    gc.collect()
    image, mask = batch
    for m in mask:
        if m.shape != (350, 525):
            m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
        valid_masks.append(m)

    for j, probability in enumerate(output):
        if probability.shape != (350, 525):
            probability = cv2.resize(probability, dsize=(
                525, 350), interpolation=cv2.INTER_LINEAR)
        probabilities[i * 4 + j, :, :] = probability

gc.collect()


def sigmoid(x): return 1/(1+np.exp(-x))


class_params = {}
for class_id in range(4):
    print(class_id)
    attempts = []
    for t in range(0, 100, 5):
        t /= 100
        for ms in tqdm.tqdm([0, 100, 1200, 5000, 10000],
                            desc=f'{class_id+1}/4; {t}/100'):
            masks = []
            for i in tqdm.tqdm(range(class_id, len(probabilities), 4)):
                probability = probabilities[i]
                predict, num_predict = post_process(
                    sigmoid(probability), t, ms)
                masks.append(predict)
            gc.collect()
            d = []
            for i, j in zip(masks, valid_masks[class_id::4]):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                else:
                    d.append(dice(i, j))
            gc.collect()
            attempts.append((t, ms, np.mean(d)))
    gc.collect()
    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (np.float(best_threshold), np.int(best_size))


with open(f'class_params_{args.model.lower()}.json', 'w') as fp:
    json.dump(class_params, fp)
