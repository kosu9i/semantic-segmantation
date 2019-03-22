#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from models.unet.model import UNet
from models.linknet.model import LinkNet
from flow.generator import train_generator, valid_generator

HERE = os.path.dirname(os.path.abspath(__file__)) + '/' 

def train(model_type):

    print('Model "{}" is selected.'.format(model_type))

    df_train = pd.read_csv(HERE + '../../../data/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    input_shape = (128, 128)
    batch_size = 32

    if model_type == 'unet':
        model = UNet(input_shape=(input_shape[0], input_shape[1], 3))
    elif model_type == 'linknet':
        model = LinkNet(input_shape=(input_shape[0], input_shape[1], 3))
    else:
        raise RuntimeError('Model "{}" is not found.'.format(model_type))

    model.build()
    print(model.model.summary())

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath='weights/best_weights_{}.hdf5'.format(input_shape[0]),
                                 save_best_only=True,
                                 save_weights_only=True),
                 TensorBoard(log_dir='logs/input_size_{}'.format(input_shape[0]))]

    train_gen = train_generator(
            ids_train_split=ids_train_split,
            train_dir=os.path.abspath(HERE + '../../../data/train/'),
            train_masks_dir=os.path.abspath(HERE + '../../../data/train_masks/'),
            batch_size=batch_size,
            input_shape=input_shape
        )

    valid_gen = valid_generator(
            ids_valid_split=ids_valid_split,
            valid_dir=os.path.abspath(HERE + '../../../data/train/'),
            valid_masks_dir=os.path.abspath(HERE + '../../../data/train_masks/'),
            batch_size=batch_size,
            input_shape=input_shape
        )

    model.fit(
        train_generator=train_gen,
        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
        epochs=100,
        verbose=2,
        callbacks=callbacks,
        validation_data=valid_gen,
        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size))
    )


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_type',
                        choices=['unet', 'linknet'],
                        default='unet')
    return parser.parse_args()

def main():
    args = arg_parse()
    train(model_type=args.model_type)

if __name__ == '__main__':
    main()

