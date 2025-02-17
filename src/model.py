import tensorflow as tf
from tensorflow.keras import layers, regularizers
import librosa
import numpy as np

def custom_accuracy(y_true, y_pred):
    threshold = 0.25
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def build_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), input_shape=(74, None, 1), kernel_regularizer=regularizers.l1(1e-5), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3),  kernel_regularizer=regularizers.l1(1e-5), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l1(1e-5), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
 
        layers.GlobalMaxPooling2D(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dense(37, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[custom_accuracy,
                                                                            tf.keras.metrics.Precision(name="precision"),
                                                                            tf.keras.metrics.Recall(name="recall")])
    
    return model




