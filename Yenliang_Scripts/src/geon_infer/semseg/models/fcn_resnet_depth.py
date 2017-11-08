from keras.models import Model
from keras.layers import (
    Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf
from .resnet50 import ResNet50
import pdb


def make_fcn_resnet_depth(input_shape, nb_labels, use_pretraining, freeze_base):
    nb_rows, nb_cols, _ = input_shape

    input_tensor = Input(shape=input_shape)
    weights = 'imagenet' if use_pretraining else None

    model = ResNet50(
        include_top=False, weights=weights, input_tensor=input_tensor, input_shape = input_shape)

    if freeze_base:
        for layer in model.layers:
            layer.trainable = False

    x32 = model.get_layer('act3d').output
    x16 = model.get_layer('act4f').output
    x8 = model.get_layer('act5c').output
   
    def resize_bilinear(images):
        return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

    # 1x1 convolution
    c32 = Conv2D(1, (1, 1), name='conv_labels_32_depth')(x32)
    c16 = Conv2D(1, (1, 1), name='conv_labels_16_depth')(x16)
    c8 = Conv2D(1, (1, 1), name='conv_labels_8_depth')(x8)
    
    # bilinear interploation 
    r32 = Lambda(resize_bilinear, name='resize_labels_32_depth')(c32)
    r16 = Lambda(resize_bilinear, name='resize_labels_16_depth')(c16)
    r8 = Lambda(resize_bilinear, name='resize_labels_8_depth')(c8)

    m = Add(name='merge_labels_depth')([r32, r16, r8])
    
    #x = Reshape((nb_rows * nb_cols, 1))(m)
    #x = Activation('sigmoid')(x)
    #x = Reshape((nb_rows, nb_cols, 1))(x)

    model = Model(inputs=input_tensor, outputs=m) 

    return model
