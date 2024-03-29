from models.linknet_attention import LinkNetGated
import json
from catalyst.utils import get_device
import ttach as tta
import torch
import argparse
from dataloader.loader import CloudDataset
from torch.utils.data import DataLoader
from catalyst.dl.runner import SupervisedRunner

import tqdm
import cv2
import pandas as pd
from utils.utils import (get_preprocessing, post_process,
                         get_validation_augmentation)
import gc
import segmentation_models_pytorch as smp
from utils import sigmoid
from dataloader.loader import mask2rle


parser = argparse.ArgumentParser("PyTorch Segmentation Pipeline")
args = parser.add_argument('-F', '--fold', default=1, type=int)
args = parser.add_argument('-M', '--model', default='smpunet', type=str)
args = parser.add_argument('-A', '--encoder', default='resnet18', type=str)
args = parser.add_argument('--out-name', default='sub_no_tta', type=str)
args = parser.add_argument('--tta', default=False, type=bool)
args = parser.add_argument('--bs', default=1, type=int)
args = parser.add_argument('--size', default=320, type=int)
args = parser.parse_args()
path = '../data/cloud_data'
logdir = f'./logs/{args.model}/fold_{args.fold}'
sub = pd.read_csv(f'{path}/sample_submission.csv')
preprocessing_fn = smp.encoders.get_preprocessing_fn(
    encoder_name=args.encoder, pretrained='imagenet')

test_ids = sub['Image_Label'].apply(
    lambda x: x.split('_')[0]).drop_duplicates().values
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

test_dataset = CloudDataset(df=sub,
                            image_size=(args.size, args.size*2),
                            path=path,
                            datatype='test',
                            preload=False,
                            img_ids=test_ids,
                            filter_bad_images=True,
                            transforms=get_validation_augmentation(
                                (args.size,
                                 args.size*2)),
                            preprocessing=get_preprocessing(preprocessing_fn))
test_loader = DataLoader(test_dataset,
                         batch_size=args.bs,
                         shuffle=False,
                         num_workers=4)

loaders = {"test": test_loader}
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
encoded_pixels = []

model = models[args.model.lower()][0](**models[args.model.lower()][1]).cuda()

class_params = json.load(
    open(f"./class_params_{args.model.lower()}.json", "r"))
""" {"0": [0.5, 10000], "1": [0.5, 10000],
                "2": [0.5, 10000], "3": [0.5, 10000]} """
image_id = 0

checkpoint = torch.load(
    f'./logs/{args.model}/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
gc.collect()


if args.tta:
    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode="sum")
else:
    tta_model = model  # tta.SegmentationTTAWrapper(
# model, tta.aliases.flip_transform(), merge_mode="mean")
runner = SupervisedRunner(
    model=tta_model,
    device=get_device())
for i, test_batch in enumerate(tqdm.tqdm(loaders['test'])):
    test_batch = test_batch[0].cuda()
    runner_out = runner.predict_batch(
        {"features": test_batch})['logits']
    gc.collect()
    for i, batch in enumerate(runner_out):
        for probability in batch:
            probability = probability.cpu().detach().numpy()
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(
                    525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(
                sigmoid(probability),
                class_params[f"{image_id % 4}"][0],
                class_params[f"{image_id % 4}"][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1


del model
gc.collect()
torch.cuda.empty_cache()

assert len(encoded_pixels) == 14792
sub['EncodedPixels'] = encoded_pixels
sub.to_csv(f'{args.out_name}.csv', columns=['Image_Label',
                                            'EncodedPixels'],
           index=False)
