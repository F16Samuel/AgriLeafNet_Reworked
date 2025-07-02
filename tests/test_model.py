# tests/test_model.py

from models.agrileafnet import build_agrileafnet

def test_model_structure():
    model = build_agrileafnet(input_shape=(224, 224, 3), num_classes=15)
    model.build((None, 224, 224, 3))
    assert model.output_shape == (None, 15)
    assert model.count_params() > 1_000_000
