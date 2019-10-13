import os
import cv2
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset


def get_img(x, path: str = '../data/cloud_data', folder: str = 'train_images'):
    """
    Return image based on image name and folder.
    """
    data_folder = f"{path}/{folder}"
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg',
              shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class CloudDataset(Dataset):

    def __init__(self, path: str = '',
                 df: pd.DataFrame = None,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip()]),
                 preprocessing=None,
                 preload: bool = False,
                 image_size: tuple = (320, 640),
                 augmentation: str = 'default',
                 filter_bad_images: bool = False):
        """

        Args:
            path: path to data
            df: dataframe with data
            datatype: train|valid|test
            img_ids: list of imagee ids
            transforms: albumentation transforms
            preprocessing: preprocessing if necessary
            preload: whether to preload data
            image_size: image size for resizing
            augmentation: name of augmentation settings
            filter_bad_images: to filter out bad images
        """

        self.df = df
        self.path = path
        self.datatype = datatype if datatype == 'test' else 'train'
        if self.datatype != 'test':
            self.data_folder = f"{path}/train_images/train_images"
        else:
            self.data_folder = f"{path}/test_images/test_images"
        self.img_ids = img_ids
        # list of bad images from discussions
        self.bad_imgs = ['046586a.jpg', '1588d4c.jpg',
                         '1e40a05.jpg', '41f92e5.jpg',
                         '449b792.jpg', '563fc48.jpg',
                         '8bd81ce.jpg', 'c0306e5.jpg',
                         'c26c635.jpg', 'e04fea3.jpg',
                         'e5f2f24.jpg', 'eda52f2.jpg',
                         'fa645da.jpg']
        if filter_bad_images:
            self.img_ids = [i for i in self.img_ids if i not in self.bad_imgs]
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.dir_name = f"{self.path}/preload_{augmentation}_{image_size[0]}_{image_size[1]}"

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):
        """
        Saves train images with augmentations, to speed up training.

        Returns:

        """
        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_mask.npy", mask)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded and self.datatype != 'valid':
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")
            mask = np.load(f"{self.dir_name}/{image_name}_mask.npy")

        else:
            mask = make_mask(self.df, image_name)
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']

        return img, mask

    def __len__(self):
        return len(self.img_ids)
