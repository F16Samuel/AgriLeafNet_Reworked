import tensorflow as tf
import albumentations as A

def try_imports():
    try:
        import tensorflow as tf
        print(f"[✓] TensorFlow imported successfully — version: {tf.__version__}")
    except ImportError as e:
        print(f"[✗] TensorFlow import failed: {e}")

    try:
        import albumentations as A
        print(f"[✓] Albumentations imported successfully — version: {A.__version__}")
    except ImportError as e:
        print(f"[✗] Albumentations import failed: {e}")