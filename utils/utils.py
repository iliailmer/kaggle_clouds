import matplotlib.pyplot as plt
import albumentations as albu
import gc
import torch
import cv2
import numpy as np


def visualize(image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i +
                1].set_title(f'Original mask {class_dict[i]}',
                             fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i +
                1].set_title(f'Transformed mask {class_dict[i]}',
                             fontsize=fontsize)


def visualize_with_raw(image, mask, original_image=None,
                       original_mask=None, raw_image=None, raw_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i +
            1].set_title(f'Original mask {class_dict[i]}',
                         fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i +
            1].set_title(f'Raw predicted mask {class_dict[i]}',
                         fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(
            f'Predicted mask with processing {class_dict[i]}',
            fontsize=fontsize)


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `visualize` function.
    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented['image']
    mask_flipped = augmented['mask']
    visualize(image_flipped, mask_flipped,
              original_image=image, original_mask=mask)


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation(p=0.5, size=(320, 640)):
    train_transform = [
        albu.Resize(*size),
        albu.HorizontalFlip(p=p),
        albu.ShiftScaleRotate(scale_limit=0.5,
                              rotate_limit=15,
                              shift_limit=0.1,
                              border_mode=0,
                              p=p),
        albu.GridDistortion(p=p),
        albu.OpticalDistortion(distort_limit=0.1,
                               shift_limit=0.5,
                               p=p),
        # albu.Cutout(p=p),
        albu.OneOf([albu.Blur(p=p),
                    albu.GaussianBlur(p=p)], p=p),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(size[0], size[1])
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if preprocessing_fn is not None:
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    else:
        _transform = [
            albu.Normalize(),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    return albu.Compose(_transform)


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with
    fewer pixels than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def training_loop(epoch, num_epochs, model, trainloader,
                  criterion, scheduler,
                  optimizer, tqdm, acc_steps=1):
    assert acc_steps > 0, f"Accumulation steps is <0 : {acc_steps}"
    pbar = tqdm(trainloader)
    model.train()
    running_loss = 0.
    i = 0
    for (x, mask) in pbar:
        y_pred = model(x.cuda())  # torch.softmax(, dim=1)
        loss = criterion(y_pred,
                         mask.cuda())
        loss = loss / acc_steps
        running_loss += (loss.item())/len(trainloader)
        loss.backward()
        if (i + 1) % acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        pbar.set_postfix({'Epoch': f'{epoch+1}/{num_epochs}',
                          'current_loss': f'{loss.item() * acc_steps:.2f}'})
        gc.collect()
        i += 1
    scheduler.step()

    pbar.set_description(f'{running_loss:.2f}')


def validation_loop(epoch, num_epochs, model, valloader, criterion,
                    tqdm, metric):
    model.eval()
    pbar = tqdm(valloader)
    running_loss = 0.
    _metric = 0.
    with torch.no_grad():
        for (x, mask) in pbar:
            y_pred = model(x.cuda())  # torch.softmax(, dim=1)
            loss = criterion(y_pred,
                             mask.cuda())
            running_loss += (loss.item())/len(valloader)
            _metric += metric(y_pred, mask.cuda())
        pbar.set_postfix({'Epoch': f'{epoch+1}/{num_epochs}',
                          'running_loss': f'{running_loss:.2f}'})
        gc.collect()
    return running_loss, 1.-_metric


def freeze(modules):
    assert isinstance(modules, list), "Modules must be passed as a list."
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(False)


def sigmoid(x): return 1/(1+np.exp(-x))
