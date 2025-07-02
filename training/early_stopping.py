# training/early_stopping.py

from tensorflow.keras.callbacks import Callback

class CustomEarlyStopping(Callback):
    def __init__(self, patience=10, monitor='val_loss', mode='min'):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        improvement = (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best)

        if improvement:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                self.model.stop_training = True
