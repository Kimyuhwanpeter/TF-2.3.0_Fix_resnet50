# -*- coding: utf-8 -*-
import tensorflow as tf

def res_block(x, filters, weight_decay, i=0):

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=2 if i == 0 else 1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters*4,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)

    if i == 0:
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.ReLU()(h + x)    
    
    return h

def gender_model(input_shape=(224, 224, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 224 x 224 x 64
    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")(h)   # 112 x 112 x 64

    for i in range(3):
        h = res_block(h, 64, weight_decay, i=i)    # 56 x 56 x 256

    for i in range(4):
        h = res_block(h, 128, weight_decay, i=i)   # 28 x 28 x 512

    for i in range(6):
        h = res_block(h, 256, weight_decay, i=i)   # 14 x 14 x 1024

    for i in range(3):
        h = res_block(h, 512, weight_decay, i=i)   # 7 x 7 x 2048

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)