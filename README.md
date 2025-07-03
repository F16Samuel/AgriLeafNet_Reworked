# 🌿 AgriLeafNet - Leaf Disease Detection Using Modified ResNet (TensorFlow)

AgriLeafNet is a modular deep learning pipeline to detect plant leaf diseases using a modified ResNet50 architecture with integrated explainability (Grad-CAM, LIME, t-SNE).

## 📁 Project Structure

```
AgriLeafNet/
├── configs/             # YAML configs for training
├── data/                # TF Datasets, Albumentations
├── models/              # ResNet50 + LoRA + Feature Heads
├── training/            # Training & evaluation
├── explainability/      # GradCAM, LIME, t-SNE
├── utils/               # Logging, metrics, seeding
├── outputs/             # Checkpoints, logs, visualizations
├── tests/               # Pytest-based unit tests
├── notebooks/           # Analysis notebooks
├── main.py              # CLI runner
├── requirements.txt
└── Dockerfile
```

## 🚀 Getting Started

### Setup (Python 3.10+)
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python main.py --mode train --config configs/train_config.yaml
```

### Evaluate Model
```bash
python main.py --mode test --model_path outputs/checkpoints/agrileafnet_best.keras
```

### Run Explainability
```bash
# GradCAM
python main.py --mode explain --explain_type gradcam --model_path <path> --image_path <image.jpg>

# LIME
python main.py --mode explain --explain_type lime --model_path <path> --image_path <image.jpg>

# t-SNE
python main.py --mode explain --explain_type tsne --model_path <path>
```

### Run Tests
```bash
pytest tests/ --tb=short -v
```

## 🧠 Features

- ResNet50 backbone with:
  - LoRA-style adapters
  - Skip connections
  - Custom feature heads
- Albumentations for TF
- Cosine Annealing + EarlyStopping
- Explainability: GradCAM, LIME, t-SNE
- Pytest test suite

## 🐳 Docker Support

```bash
docker build -t agrileafnet .
docker run --gpus all -v $PWD:/app agrileafnet
```

## 🔬 Dataset

- **PlantVillage (Kaggle)** – Ensure images are placed inside `data/PlantVillage`.

## 📄 License

Apache License 2.0.
