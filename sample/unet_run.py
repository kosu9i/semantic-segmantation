#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../model/')

from unet import UNet


def main():
    model = UNet(
        input_shape=(1024, 1024, 3),
        num_filters={
            'down': [8, 16, 32, 64, 128, 256, 512],
            'center': 1024,
            'up': [512, 256, 128, 64, 32, 16, 8]
        } 
    )
    model.build()
    print(model.model.summary())

    model = UNet(input_shape=(128, 128, 3))
    model.build()
    print(model.model.summary())


if __name__ == '__main__':
    main()
