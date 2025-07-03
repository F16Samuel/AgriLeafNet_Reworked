# models/lora_utils.py

import tensorflow as tf

class LoRAAdapter(tf.keras.layers.Layer):
    def __init__(self, units, rank=8, alpha=1.0, name="lora_adapter"):
        super().__init__(name=name)
        self.rank = rank
        self.alpha = alpha
        self.dense_down = tf.keras.layers.Dense(rank, use_bias=False, trainable=True)
        self.dense_up = tf.keras.layers.Dense(units, use_bias=False, trainable=True)

    def call(self, inputs):
        lora_out = self.dense_up(self.dense_down(inputs))
        return inputs + self.alpha * lora_out