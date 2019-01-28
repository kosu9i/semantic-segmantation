# coding: utf-8

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

from .losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

class UNet(object):

    MAX_FILTER_NUM = 1024

    def __init__(self,
                 input_shape=(128, 128, 3),
                 num_classes=1,
                 kernel_size=(3, 3),
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
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_filters = num_filters

        self._validate_params()

        self.callbacks = []

    def _validate_params(self):
        # TODO
        pass

    def build(self):
        inputs = Input(shape=self.input_shape)
        prev_output = inputs

        # add down layer
        down_list = []
        for num_filter in self.num_filters['down']:
            down = Conv2D(num_filter, self.kernel_size, padding=self.padding)(prev_output)
            down = BatchNormalization()(down)
            down = Activation('relu')(down)
            down = Conv2D(num_filter, self.kernel_size, padding=self.padding)(down)
            down = BatchNormalization()(down)
            down = Activation('relu')(down)
            down_pool = MaxPooling2D((2, 2), strides=self.strides)(down)
            down_list.append(down)
            prev_output = down_pool 

        # add center layer
        center = Conv2D(self.num_filters['center'], self.kernel_size, padding=self.padding)(prev_output)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(self.num_filters['center'], self.kernel_size, padding=self.padding)(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        prev_output = center

        # add up layer
        for i, num_filter in enumerate(self.num_filters['up'], start=1):
            up = UpSampling2D((2, 2))(prev_output)
            up = concatenate([down_list[-i], up], axis=3)
            up = Conv2D(num_filter, self.kernel_size, padding=self.padding)(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            up = Conv2D(num_filter, self.kernel_size, padding=self.padding)(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            up = Conv2D(num_filter, self.kernel_size, padding=self.padding)(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            prev_output = up
             
        # add output layer
        classify = Conv2D(self.num_classes, (1, 1), activation='sigmoid')(prev_output)

        self.model = Model(inputs=inputs, outputs=classify)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    def fit(self, train_generator, steps_per_epoch, epochs, verbose, callbacks, validation_data, validation_steps):
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps
        )
