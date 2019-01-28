# coding: utf-8

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

class UNet(object):

    MAX_FILTER_NUM = 1024

    def __init__(self,
                 input_shape=(128, 128, 3),
                 num_classes=1,
                 kernel=(3, 3),
                 strides=(2, 2),
                 padding='same',
                 num_filters={
                    'down': [64, 128, 256, 512],
                    'center': 1024,
                    'up': [512, 256, 128, 64]
                    }
                 ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.num_filters = num_filters

    def _create_filter_list(self):
        num_filter = self.input_shape[0]
        filter_list = []
        i = 1
        while True:
            num_filter = num_filter / 2
            if num_filter < 8:
                break
            filter_list.append(int(self.MAX_FILTER_NUM / (2 ** i)))
            i += 1
        return filter_list


    def build(self):
        filter_list = self._create_filter_list()

        inputs = Input(shape=self.input_shape)

        prev_output = inputs

        down_list = []
        for filter_num in reversed(filter_list):
            down = Conv2D(filter_num, (3, 3), padding='same')(prev_output)
            down = BatchNormalization()(down)
            down = Activation('relu')(down)
            down = Conv2D(filter_num, (3, 3), padding='same')(down)
            down = BatchNormalization()(down)
            down = Activation('relu')(down)
            down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)
            down_list.append(down)
            prev_output = down_pool 

        center = Conv2D(self.MAX_FILTER_NUM, (3, 3), padding='same')(prev_output)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(self.MAX_FILTER_NUM, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)

        prev_output = center

        for i, filter_num in enumerate(filter_list, start=1):
            up = UpSampling2D((2, 2))(prev_output)
            up = concatenate([down_list[-i], up], axis=3)
            up = Conv2D(filter_num, (3, 3), padding='same')(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            up = Conv2D(filter_num, (3, 3), padding='same')(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            up = Conv2D(filter_num, (3, 3), padding='same')(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            prev_output = up
           
        classify = Conv2D(self.num_classes, (1, 1), activation='sigmoid')(prev_output)

        self.model = Model(inputs=inputs, outputs=classify)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

