import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✔️ Memory growth enabled for GPU")
    except RuntimeError as e:
        print("❌ Memory growth setting failed:", e)