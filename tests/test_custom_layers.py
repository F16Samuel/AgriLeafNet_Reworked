# tests/test_custom_layers.py

import tensorflow as tf
from models.custom_layers import FeatureHead, ResidualBlock
from models.lora_utils import LoRAAdapter

def test_feature_head_output_shape():
    layer = FeatureHead(128)
    output = layer(tf.random.normal([4, 512]), training=True)
    assert output.shape == (4, 128)

def test_residual_block_shape():
    block = ResidualBlock(256)
    out = block(tf.random.normal([4, 256]), training=True)
    assert out.shape == (4, 256)

def test_lora_adapter():
    lora = LoRAAdapter(units=512, rank=4)
    out = lora(tf.random.normal([2, 512]))
    assert out.shape == (2, 512)
