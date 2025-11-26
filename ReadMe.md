# 🦴 Bone Fracture Detection System

A comprehensive deep learning solution for detecting and segmenting bone fractures in X-ray images.

## 📋 Project Overview

This project implements a semantic segmentation CNN model with the following features:

- **Architecture**: U-Net based with ResNet50 backbone
- **Task**: Semantic segmentation of fracture regions + bounding box detection
- **Optimization**: GPU acceleration with MPS (Metal Performance Shaders) for Mac
- **Inference**: Real-time prediction with Streamlit interface
- **Metrics**: Dice coefficient, IoU, Sensitivity, Specificity

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create project directory
mkdir bone_fracture_detection
cd bone_fracture_detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy pandas scikit-learn matplotlib seaborn
pip install kaggle streamlit albumentations tensorboard tqdm
```

### 2. Kaggle API Setup

```bash
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token" (downloads kaggle.json)
# 3. Set up authentication:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Verify
kaggle datasets list
```

### 3. Run Complete Pipeline

```bash
# Run full pipeline (download → preprocess → train)
python3 main.py --full

# Or run individually:
python3 main.py --download      # Download dataset
python3 main.py --preprocess    # Preprocess dataset
python3 main.py --train         # Train model
python3 main.py --streamlit     # Launch Streamlit app
```

## 📁 Project Structure

```
bone_fracture_detection/
├── model.py                 # CNN model architecture
├── preprocess.py            # Data preprocessing pipeline
├── train.py                 # Training script with metrics
├── app.py                   # Streamlit inference interface
├── main.py                  # Master pipeline script
├── setup.sh                 # Setup script
├── checkpoints/
│   └── best_model.pth      # Trained model checkpoint
├── processed_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── training_metrics.png    # Loss and metric plots
└── training_history.json   # Training statistics
```

## 🔧 Detailed Workflow

### Step 1: Data Preprocessing

The preprocessing script (`preprocess.py`) performs:

- **Polygon-to-Mask Conversion**: Converts YOLOv8 polygon annotations to binary segmentation masks
- **Image Standardization**: Resizes images to 640×640 maintaining aspect ratio with padding
- **Data Organization**: Structures data for train/val/test splits
- **Statistics Calculation**: Generates class distribution and image statistics
- **Visualization**: Creates sample visualizations

```bash
python3 preprocess.py
```

**Output:**
- `processed_dataset/` directory with organized train/val/test splits
- `visualization_samples.png` showing preprocessed samples
- Statistics summary in console

### Step 2: Model Training

The training script (`train.py`) includes:

- **Architecture**: Custom U-Net with ResNet50 encoder
- **Loss Function**: Combined Dice + BCE loss (0.7:0.3 weighting)
- **Optimization**: AdamW optimizer with cosine annealing scheduler
- **Metrics Tracking**: Dice, IoU, Sensitivity, Specificity per epoch
- **Early Stopping**: Saves best model based on validation Dice coefficient

```bash
python3 train.py
```

**Configuration:**
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 30
- Data augmentation: Flips, brightness, contrast adjustments

**Outputs:**
- `checkpoints/best_model.pth` - Best model weights
- `training_metrics.png` - Loss and metric plots
- `training_history.json` - Training statistics

### Step 3: Evaluation Metrics

The model tracks multiple metrics:

- **Dice Coefficient**: Measures overlap between predicted and true masks (0-1, higher is better)
- **IoU (Intersection over Union)**: Standard segmentation metric
- **Sensitivity**: True positive rate for fracture detection
- **Specificity**: True negative rate for normal regions
- **Loss**: Combined Dice + BCE loss

These are plotted and saved after training.

### Step 4: Inference & Visualization

Launch the Streamlit app for interactive inference:

```bash
streamlit run app.py
```

**Features:**
- Upload X-ray images
- Real-time inference with MPS acceleration
- Segmentation mask visualization
- Bounding box detection with confidence scores
- Download results as images
- Confidence threshold adjustment

## 🎯 Model Architecture

### Encoder (ResNet50)
- Layer 0: 64 channels
- Layer 1: 64 channels (1/4 resolution)
- Layer 2: 128 channels (1/8 resolution)
- Layer 3: 256 channels (1/16 resolution)
- Layer 4: 512 channels (1/32 resolution)

### Bottleneck
- Double convolution: 512 → 1024 channels

### Decoder (Progressive Upsampling)
- Skip connections from each encoder level
- DoubleConv blocks after concatenation
- Bilinear upsampling with learned refinement

### Output Head
- Final convolution: 32 → 16 → 1 channel
- Sigmoid activation for binary segmentation

## 💻 Mac M4 Pro Optimization

This project is optimized for MacBook M4 Pro:

```python
# Automatic MPS detection
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Efficient settings for Mac
- Batch size: 8 (reduces memory)
- Pin memory: True
- Num workers: 4
- Target image size: 640×640
```

**Expected Performance:**
- Training time: ~2-3 hours for 30 epochs
- Inference time: ~50-100ms per image
- Memory usage: ~4-6GB

## 📊 Training Monitoring

During training, monitor:

1. **Loss Convergence**: Train and validation loss should decrease
2. **Dice Score**: Should increase over epochs (target: >0.85)
3. **IoU Score**: Should track with Dice coefficient
4. **Sensitivity/Specificity**: Balance between detecting fractures and avoiding false positives

## 🔍 Inference Output

For each input image, the model outputs:

1. **Segmentation Mask**: Binary mask showing fracture region
2. **Bounding Box**: Coordinates and dimensions (x, y, width, height)
3. **Confidence Score**: Model confidence in fracture presence (0-1)
4. **Label**: "Fracture" with confidence percentage

## 🎓 Educational Contributions

This project demonstrates:

- Custom U-Net architecture implementation
- Transfer learning with pre-trained ResNet50
- Polygon-to-mask conversion for segmentation tasks
- Combined loss functions for multi-task learning
- Data augmentation strategies
- PyTorch training pipeline with validation
- Streamlit application development
- Model evaluation with multiple metrics
- GPU optimization for Apple Silicon

## 📝 Key Customizations for Learning

The code includes detailed comments explaining:

- Model architecture choices (why U-Net with skip connections)
- Loss function design (Dice + BCE combination)
- Data preprocessing (aspect ratio preservation with padding)
- Training optimization (learning rate scheduling, gradient clipping)
- Metric calculation (true positive, false positive handling)

## 🐛 Troubleshooting

### MPS Not Available
```python
# Check if MPS is available
python3 -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch
pip install --upgrade torch torchvision
```

### Out of Memory
```python
# Reduce batch size in train.py
batch_size: 4  # or 2
```

### Kaggle Authentication Issues
```bash
# Verify credentials
cat ~/.kaggle/kaggle.json
# Should show: {"username":"...", "key":"..."}
```

### Model Not Found for Inference
```bash
# Train model first
python3 main.py --train

# Check checkpoint exists
ls checkpoints/best_model.pth
```

## 📚 References

- **U-Net**: Ronneberger et al., 2015
- **ResNet**: He et al., 2015
- **Dice Loss**: Milletari et al., 2016
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Streamlit Docs**: https://docs.streamlit.io/

## 🤝 Contributing

Suggestions for improvements:

- Add more augmentation techniques (rotation, elastic deformation)
- Implement multi-class classification (fracture types)
- Add attention mechanisms to the decoder
- Create ensemble models for robustness
- Add explainability features (saliency maps, GradCAM)

## 📄 License

This project is for educational purposes.

## ✨ Acknowledgments

- Kaggle dataset provider: pkdarabi
- PyTorch community for excellent documentation

---

**Last Updated**: 2025
**Tested On**: MacBook M4 Pro, macOS Ventura/Sonoma
