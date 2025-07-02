# tests/test_training.py

from training.train import train

def test_training_loop_runs(tmp_path):
    model, history = train("configs/train_config.yaml")
    assert len(history.history["loss"]) >= 1
    assert "val_accuracy" in history.history