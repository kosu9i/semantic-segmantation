import cv2
import numpy as np
from preprocess.augmentation import random_hue_saturation_value, random_shift_scale_rotate, random_flip

def train_generator(ids_train_split, train_dir, train_masks_dir, batch_size, input_shape):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread('{}/{}.jpg'.format(train_dir, id))
                img = cv2.resize(img, input_shape)
                mask = cv2.imread('{}/{}_mask.png'.format(train_masks_dir, id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, input_shape)
                img = random_hue_saturation_value(img,
                                                  hue_shift_limit=(-50, 50),
                                                  sat_shift_limit=(-5, 5),
                                                  val_shift_limit=(-15, 15))
                img, mask = random_shift_scale_rotate(img, mask,
                                                      shift_limit=(-0.0625, 0.0625),
                                                      scale_limit=(-0.1, 0.1),
                                                      rotate_limit=(-0, 0))
                img, mask = random_flip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator(ids_valid_split, valid_dir, valid_masks_dir, batch_size, input_shape):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread('{}/{}.jpg'.format(valid_dir, id))
                img = cv2.resize(img, input_shape)
                mask = cv2.imread('{}/{}_mask.png'.format(valid_masks_dir, id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, input_shape)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch



