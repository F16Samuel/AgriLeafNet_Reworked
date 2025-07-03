# models/lora_utils.py

import tensorflow as tf

import tensorflow as tf

class LoRAAdapter(tf.keras.layers.Layer):
    def __init__(self, units, rank=8, alpha=1.0, residual=True, name="lora_adapter"):
        super().__init__(name=name)
        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.residual = residual

        self.dense_down = tf.keras.layers.Dense(rank, use_bias=False, trainable=True)
        self.dense_up = tf.keras.layers.Dense(units, use_bias=False, trainable=True)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        if self.residual and self.input_dim != self.units:
            raise ValueError(
                f"LoRAAdapter: input dim ({self.input_dim}) must match output dim ({self.units}) if residual=True."
            )

    def call(self, inputs):
        lora_out = self.dense_up(self.dense_down(inputs))
        if self.residual:
            return inputs + self.alpha * lora_out
        else:
            return self.alpha * lora_out