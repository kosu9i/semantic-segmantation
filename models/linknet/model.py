# coding: utf-8
 
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, add
from keras.optimizers import RMSprop
from keras import backend as K
from keras.regularizers import l2
 
from models.unet.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
 
class LinkNet(object):
 
    # TODO: 検証時に変更したいパラメータを引数にとるようにしたい
    def __init__(self,
                 input_shape=(128, 128, 3),
                 num_classes=1,
                 ):
        self.input_shape = input_shape
        self.num_classes = num_classes
 
    def build(self):
 
        inputs = Input(shape=self.input_shape)
 
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(x)
 
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
 
        encoder_1 = self._encoder_block(input_tensor=x, m=64, n=64)
 
        encoder_2 = self._encoder_block(input_tensor=encoder_1, m=64, n=128)
 
        encoder_3 = self._encoder_block(input_tensor=encoder_2, m=128, n=256)
 
        encoder_4 = self._encoder_block(input_tensor=encoder_3, m=256, n=512)
 
        decoder_4 = self._decoder_block(input_tensor=encoder_4, m=512, n=256)
 
        decoder_3_in = add([decoder_4, encoder_3])
        decoder_3_in = Activation('relu')(decoder_3_in)
 
        decoder_3 = self._decoder_block(input_tensor=decoder_3_in, m=256, n=128)
 
        decoder_2_in = add([decoder_3, encoder_2])
        decoder_2_in = Activation('relu')(decoder_2_in)
 
        decoder_2 = self._decoder_block(input_tensor=decoder_2_in, m=128, n=64)
 
        decoder_1_in = add([decoder_2, encoder_1])
        decoder_1_in = Activation('relu')(decoder_1_in)
 
        decoder_1 = self._decoder_block(input_tensor=decoder_1_in, m=64, n=64)
 
        x = UpSampling2D((2, 2))(decoder_1)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
 
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=self.num_classes, kernel_size=(2, 2), padding="same")(x)
 
        model = Model(inputs=inputs, outputs=x)
 
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    def _shortcut(self, input, residual):
        """Adds a shortcut between input and residual block and merges them with "sum"
        """
        # Expand channels of shortcut to match residual.
        # Stride appropriately to match residual (width, height)
        # Should be int if network architecture is correctly configured.
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]
     
        shortcut = input
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3],
                              kernel_size=(1, 1),
                              strides=(stride_width, stride_height),
                              padding="valid",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(0.0001))(input)
     
        return add([shortcut, residual])
     
    def _encoder_block(self, input_tensor, m, n):
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
     
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
     
        added_1 = self._shortcut(input_tensor, x)
     
        x = BatchNormalization()(added_1)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
     
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
     
        added_2 = self._shortcut(added_1, x)
     
        return added_2
     
    def _decoder_block(self, input_tensor, m, n):
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=int(m/4), kernel_size=(1, 1))(x)
     
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=int(m/4), kernel_size=(3, 3), padding='same')(x)
     
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n, kernel_size=(1, 1))(x)
     
        return x

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
