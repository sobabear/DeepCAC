"""
  ----------------------------------------
    Heart segmentation - DL model arch.
  ----------------------------------------
  ----------------------------------------
  Author: AIM Harvard
  
  Python Version: 3.x
  ----------------------------------------
  
"""
import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Input
from tensorflow.keras.utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler

from functools import partial


def dice_coef(y_true, y_pred, smooth=1.):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

## ----------------------------------------
## ----------------------------------------

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)

## ----------------------------------------
## ----------------------------------------

def getUnet3d(down_steps, input_shape, pool_size=(2, 2, 2), conv_size=(3, 3, 3), initial_learning_rate=0.00001,
              mgpu=1, ext=False, drop_out=0.5):
  # Convert input_shape to tuple if it's a list
  if isinstance(input_shape, list):
    input_shape = tuple(input_shape)  # Just convert to tuple without adding channel
  
  if down_steps == 4:
    return getUnet3d_4(input_shape, pool_size=pool_size, conv_size=conv_size,
                       initial_learning_rate=initial_learning_rate, mgpu=mgpu, ext=ext)
  else:
    print('Wrong U-Net parameters specified ("down_steps")')
    return None


# 4 Down steps - MultiGPU
def getUnet3d_4(input_shape, pool_size, conv_size, initial_learning_rate, mgpu, ext=False):
  # Create the model
  strategy = None
  if mgpu > 1:
    print('Compiling multi GPU model...')
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
  
  if strategy:
    with strategy.scope():
      model = create_unet_model(input_shape, pool_size, conv_size, initial_learning_rate, ext)
  else:
    print('Compiling single GPU model...')
    model = create_unet_model(input_shape, pool_size, conv_size, initial_learning_rate, ext)
  
  return model

def create_unet_model(input_shape, pool_size, conv_size, initial_learning_rate):
  # Ensure input_shape is a tuple
  if isinstance(input_shape, list):
    input_shape = tuple(input_shape)

  inputs = Input(input_shape)
  
  # Encoder path
  conv1 = Conv3D(32, conv_size, activation='relu', padding='same')(inputs)
  conv1 = Conv3D(64, conv_size, activation='relu', padding='same')(conv1)
  pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

  conv2 = Conv3D(64, conv_size, activation='relu', padding='same')(pool1)
  conv2 = Conv3D(128, conv_size, activation='relu', padding='same')(conv2)
  pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

  conv3 = Conv3D(128, conv_size, activation='relu', padding='same')(pool2)
  conv3 = Conv3D(256, conv_size, activation='relu', padding='same')(conv3)
  pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

  conv4 = Conv3D(256, conv_size, activation='relu', padding='same')(pool3)
  conv4 = Conv3D(512, conv_size, activation='relu', padding='same')(conv4)
  pool4 = MaxPooling3D(pool_size=pool_size)(conv4)

  # Bridge
  conv5 = Conv3D(256, conv_size, activation='relu', padding='same')(pool4)
  conv5 = Conv3D(512, conv_size, activation='relu', padding='same')(conv5)

  # Decoder path
  up6 = UpSampling3D(size=pool_size)(conv5)
  merge6 = concatenate([up6, conv4], axis=4)
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same')(merge6)
  conv6 = Conv3D(256, conv_size, activation='relu', padding='same')(conv6)

  up7 = UpSampling3D(size=pool_size)(conv6)
  merge7 = concatenate([up7, conv3], axis=4)
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same')(merge7)
  conv7 = Conv3D(128, conv_size, activation='relu', padding='same')(conv7)

  up8 = UpSampling3D(size=pool_size)(conv7)
  merge8 = concatenate([up8, conv2], axis=4)
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same')(merge8)
  conv8 = Conv3D(64, conv_size, activation='relu', padding='same')(conv8)

  up9 = UpSampling3D(size=pool_size)(conv8)
  merge9 = concatenate([up9, conv1], axis=4)
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same')(merge9)
  conv9 = Conv3D(64, conv_size, activation='relu', padding='same')(conv9)

  conv10 = Conv3D(1, (1, 1, 1))(conv9)
  outputs = Activation('sigmoid')(conv10)

  model = Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer=Adam(learning_rate=initial_learning_rate), 
               loss=dice_coef_loss,
               metrics=[dice_coef])
  return model
