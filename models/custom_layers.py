# models/custom_layers.py

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

class FeatureHead(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", dropout_rate=0.3, name="feature_head"):
        super().__init__(name=name)
        self.dense = layers.Dense(units, activation=activation, kernel_regularizer=regularizers.l2(1e-4))
        self.bn = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        return self.dropout(x, training=training)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation="relu", name="residual_block"):
        super().__init__(name=name)
        self.dense1 = layers.Dense(units, activation=activation)
        self.bn1 = layers.BatchNormalization()
        self.dense2 = layers.Dense(units)
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = layers.add([x, inputs])
        return self.activation(x)
