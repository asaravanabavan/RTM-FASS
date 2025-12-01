# RTM-FASS Models

This directory contains the trained PyTorch models for strike detection and classification.

## Model Files

Due to GitHub's file size limitations (100MB), the trained model files are **NOT included** in this repository.

### Available Models

- `best_model.pth` - Padwork-trained model (~145MB)
- `combined_strike_model.pth` - Combined padwork + fight model (~145MB)
- `FINAL_MODEL.pth` - Production model (~145MB)
- `strike_model.pth` - Initial baseline model
- `ultra_fast_model.pth` - Lightweight variant

## Getting Pre-trained Models

### Option 1: Download from Release
Check the [Releases](https://github.com/yourusername/RTM-FASS/releases) page for pre-trained models.

### Option 2: Train Your Own
Follow the training pipeline in `Copy of MAIN.ipynb`:

```python
# Train on padwork data
padwork_model = train_with_padwork_data()

# Process fight data
fight_data = process_coco_fight_data()

# Train combined model
final_model = train_combined_model(padwork_model, fight_data)
```

### Option 3: Use Git LFS (Optional)
If you set up Git LFS, you can track the model files:

```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
```

## Model Architecture

- **Type**: CNN-LSTM with Temporal Attention
- **Input**: 15-frame pose sequences (17 keypoints × 3 channels)
- **Output**: Strike classification (8 classes) + outcome prediction
- **Size**: ~145MB per model
- **Format**: PyTorch checkpoint (.pth)

## Model Performance

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Inference Speed**: 30+ FPS on RTX GPUs

## Training Curves

Training visualization outputs (`*.png`) are included in this repository to document model performance.
