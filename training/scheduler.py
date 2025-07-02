# training/scheduler.py

import math
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class CosineAnnealingScheduler(Callback):
    def __init__(self, initial_lr, t_max, eta_min=0, restart_mult=2):
        super().__init__()
        self.initial_lr = initial_lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.restart_mult = restart_mult
        self.epoch_since_restart = 0
        self.current_t_max = t_max

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * self.epoch_since_restart / self.current_t_max)) / 2
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_since_restart += 1
        if self.epoch_since_restart >= self.current_t_max:
            self.epoch_since_restart = 0
            self.current_t_max *= self.restart_mult
