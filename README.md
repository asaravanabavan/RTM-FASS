# RTM-FASS
# 🥊 RTM-FASS: Real-Time Muay Thai Fight Analysis & Scoring System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

> An AI-powered automated scoring system for Muay Thai fights using computer vision and deep learning to provide objective, unbiased scoring for each fighter.

**Final Year Project - University of Westminster**

---

## 🎯 Overview

### The Problem

Muay Thai judging faces significant challenges:
- ❌ **Subjective scoring** - Human bias affects fight outcomes
- ❌ **Inconsistent decisions** - Judges vary in scoring criteria
- ❌ **Limited analytics** - No detailed performance metrics for fighters
- ❌ **Training feedback gap** - Fighters lack objective improvement data

### The Solution

RTM-FASS provides an automated, unbiased fight analysis system that:
- ✅ **Eliminates human bias** - Consistent, rule-based scoring
- ✅ **Real-time analysis** - Instant performance feedback during fights
- ✅ **Comprehensive statistics** - Detailed breakdown of techniques and effectiveness
- ✅ **AI-powered detection** - Deep learning models trained on authentic Muay Thai data

---

## ✨ Key Features

### 🎥 Video Processing
- **Live & Recorded Analysis** - Process fights in real-time or from video files
- **Multi-Fighter Tracking** - Simultaneous detection and tracking of both fighters
- **Automatic ID Assignment** - Left/right fighter identification with persistent tracking
- **Occlusion Handling** - Maintains fighter identity even during brief occlusions
- **Adaptive Processing** - Handles varying lighting, angles, and camera distances

### 🦴 Computer Vision Pipeline
- **YOLOv8 Pose Estimation** - High-accuracy 17-point skeleton tracking
- **Batch Processing** - Efficient GPU-accelerated frame processing
- **Fighter Identification** - Consistent ID assignment throughout fight
- **Memory Management** - Smart caching and CUDA optimization for long videos

### 🧠 Deep Learning Architecture

#### CNN-LSTM Strike Recognition Model
- **Temporal Attention Mechanism** - Focuses on key frames in strike sequences
- **Bidirectional LSTM** - Captures past and future context for accurate classification
- **Multi-task Learning** - Simultaneous strike type and outcome prediction
- **Fighter-Specific Features** - Embedding layer for personalized detection

### 👊 Strike Detection System

Comprehensive detection of all Muay Thai techniques:

| Category | Techniques Detected |
|----------|-------------------|
| **Punches** | Jab, Cross, Hook, Uppercut |
| **Kicks** | Roundhouse, Teep (Front Kick) |
| **Elbows** | Various elbow strikes |
| **Knees** | Knee strikes including clinch knees |

**Detection Features:**
- Sequence-based detection (15-frame sequences)
- Cooldown periods to prevent duplicate detections
- Confidence thresholding for accuracy
- Strike outcome classification (hit/miss)

### 📊 Scoring Engine

**Traditional 10-Point Must System Implementation:**

#### Weighted Scoring Components:
- **Strikes (40%)** - Landed techniques with type-based weighting
  - Kicks: 1.5x multiplier
  - Knees/Elbows: 1.3x multiplier
  - Punches: 1.0x baseline
- **Aggression (25%)** - Ring control and offensive pressure
- **Defense (15%)** - Blocks and evasions
- **Technique (20%)** - Variety and execution quality

#### Strike Effectiveness Multipliers:
- Blocked: 0.2x
- Partially Blocked: 0.5x
- Clean: 1.0x
- Counter: 1.2x
- Knockdown: 2.0x

#### Advanced Analytics:
- Round-by-round scoring breakdown
- Fatigue tracking and recovery modeling
- Movement analysis (distance traveled, direction changes)
- Center control percentage
- Strike accuracy by type
- Defense rate calculations

### 📈 Real-Time Visualization

**On-Screen Overlays:**
- Live round timer and score display
- Strike indicators with fade-out effects
- Color-coded fighter identification
- Real-time statistics panel
- Strike accuracy tracking
- Most-used techniques breakdown

**Post-Fight Summary:**
- Comprehensive fight analysis report
- Round-by-round scorecards
- Fighter performance comparison
- Strike breakdown visualization
- Fight outcome prediction with confidence levels

---

## 🏗️ Architecture

### Project Structure

```
RTM-FASS/
├── src/
│   ├── config.py              # Configuration settings
│   ├── pose_detector.py       # YOLOv8 pose detection
│   ├── strike_detector.py     # Strike recognition logic
│   ├── strike_model.py        # CNN-LSTM neural network
│   ├── fight_analyzer.py      # Scoring and analysis engine
│   ├── visualization.py       # Real-time overlays and summaries
│   ├── data_processor.py      # Dataset processing utilities
│   └── train_model.py         # Model training pipeline
├── data/
│   ├── raw_videos/           # Source fight footage
│   ├── annotations/          # COCO format annotations
│   ├── strike_dataset/       # Processed padwork sequences
│   └── fight_dataset/        # Processed sparring sequences
├── models/
│   ├── best_model.pth        # Padwork-trained model
│   ├── combined_strike_model.pth  # Combined model
│   └── FINAL_MODEL.pth       # Production model
├── outputs/                  # Analysis results
└── Copy of MAIN.ipynb       # Complete pipeline notebook
```

### Data Pipeline

1. **Padwork Dataset Processing**
   - Extracts 15-frame sequences from technique demonstrations
   - YOLOv8 pose estimation on each frame
   - Normalizes and augments keypoint data
   - Generates labeled sequences for training

2. **Sparring Dataset Processing**
   - COCO annotation-based extraction
   - Multi-fighter tracking throughout videos
   - Context-aware sequence generation
   - Fighter-specific labels and outcomes

3. **Combined Training**
   - Transfers learning from padwork model
   - Fine-tunes on authentic fight data
   - Balanced class weighting
   - Early stopping and learning rate scheduling

### Model Training Features

- **Data Augmentation**
  - Temporal scaling (0.8x - 1.2x)
  - Rotation augmentation (±25°)
  - Scale variation (0.8x - 1.2x)
  - Gaussian noise injection

- **Training Optimizations**
  - Mixed precision training (AMP)
  - Gradient clipping
  - AdamW optimizer
  - ReduceLROnPlateau scheduling
  - Batch size: 512 (padwork), adaptive for fights

- **Performance Metrics**
  - Per-class accuracy tracking
  - Validation accuracy monitoring
  - Training/validation loss curves
  - Confusion matrix analysis

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
8GB+ GPU memory for training
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RTM-FASS.git
cd RTM-FASS
```

2. **Install dependencies**
```bash
pip install ultralytics==8.0.145
pip install opencv-python pandas numpy==1.24.3
pip install matplotlib tqdm torch==2.0.1 torchvision==0.15.2
pip install pycocotools
```

3. **Download YOLOv8 models**
```bash
# Models will auto-download on first run, or manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Quick Start

**Option 1: Analyze a Fight Video**

```python
from src.pose_detector import PoseDetector
from src.strike_detector import StrikeDetector
from src.fight_analyzer import FightAnalyzer
from src.visualization import FightVisualizer

# Initialize components
pose_detector = PoseDetector(model="yolov8x-pose.pt")
strike_detector = StrikeDetector(model_path="models/combined_strike_model.pth")
analyzer = FightAnalyzer()
visualizer = FightVisualizer()

# Process your fight video
# See Copy of MAIN.ipynb for complete implementation
```

**Option 2: Run Complete Pipeline (Google Colab)**

Open `Copy of MAIN.ipynb` and run cells marked with ⭐ for quick analysis, or run all cells to retrain models from scratch.

---

## 📊 Model Performance

### Supported Classes
- Jab, Cross, Hook, Uppercut
- Roundhouse, Teep
- Elbow, Knee

### Dataset Statistics
- **Padwork Sequences**: Thousands of labeled technique demonstrations
- **Fight Sequences**: Real sparring footage with COCO annotations
- **Data Augmentation**: 5x effective dataset size through augmentation

---

## 🎮 Usage Examples

### Processing a Fight Video

```python
# Run the complete analysis
results = run_combined_model(
    model_path="models/combined_strike_model.pth",
    video_path="path/to/fight.mp4"
)

# Output files generated:
# - analyzed_video.mp4 (annotated video)
# - summary_image.jpg (scorecards and stats)
# - stats.json (detailed metrics)
```

### Training Your Own Model

```python
# Process your padwork dataset
metadata_path = process_strike_dataset(
    csv_file_path="data/techniques_dataset.csv",
    video_directory="data/raw_videos/strikes"
)

# Train initial model
padwork_model = train_with_padwork_data()

# Process fight footage
fight_data = process_coco_fight_data()

# Train combined model
final_model = train_combined_model(
    padwork_model_path=padwork_model,
    fight_data_path=fight_data
)
```

---

## 🔬 Technical Details

### Pose Estimation
- **Keypoints**: 17 body landmarks (COCO format)
- **Confidence Threshold**: 0.25 for detection, 0.2 for keypoints
- **Skeleton Connections**: 19 limb connections visualized

### Strike Detection Pipeline
1. **Frame Buffer**: Maintains 15-frame rolling window
2. **Keypoint Extraction**: Extracts pose for both fighters
3. **Sequence Formation**: Creates temporal sequences
4. **Normalization**: Position normalization with confidence weighting
5. **Model Inference**: CNN-LSTM with attention prediction
6. **Cooldown**: 10-15 frame cooldown between detections

### Scoring Algorithm
- **Round Duration**: 180 seconds (configurable)
- **Total Rounds**: 5 rounds (configurable)
- **Score Range**: 7-10 points per round (10-point must system)
- **Dominant Fighter**: Calculated from cumulative scores
- **Win Prediction**: Confidence-based outcome prediction

---

## 📁 Output Files

### Analyzed Video
- Annotated with real-time overlays
- Color-coded fighter identification
- Strike indicators and statistics
- Round timer and score display

### Summary Image
- Round-by-round scorecards
- Fighter statistics comparison
- Strike breakdown by type
- Accuracy percentages
- Fight prediction

### Statistics JSON
```json
{
  "video_path": "...",
  "duration_seconds": 300,
  "fight_summary": {...},
  "fighter_stats": {
    "0": {"strikes": {...}, "defense": {...}},
    "1": {"strikes": {...}, "defense": {...}}
  },
  "time_series": [...]
}
```

---

## 🛠️ Configuration

Edit `src/config.py` for customization:

```python
# Video settings
VIDEO_CONFIGURATION = {
    'maximum_height': 720,
    'target_frames_per_second': 30,
    'skip_frames': 2
}

# Strike detection
STRIKE_CONFIGURATION = {
    'sequence_length': 15,
    'cooldown_frames': 15,
    'confidence_threshold': 0.6
}
```

---

## 🎓 Research & Development

This system was developed as a Final Year Project at the University of Westminster, combining:
- Computer Vision (YOLOv8, OpenCV)
- Deep Learning (PyTorch, CNN-LSTM)
- Sports Analytics (Traditional Muay Thai scoring)
- Real-time Processing (CUDA optimization)

### Key Innovations
1. **Temporal Attention Mechanism** for strike recognition
2. **Fighter-specific embeddings** for personalized detection
3. **Occlusion handling** with memory-based tracking
4. **Multi-task learning** for strike type and outcome
5. **Comprehensive scoring engine** matching professional judging criteria

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional strike types (spinning techniques, superman punch)
- Clinch work detection and scoring
- Multi-angle camera fusion
- Real-time streaming support
- Mobile deployment optimization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **University of Westminster** - Academic support
- **Ultralytics** - YOLOv8 framework
- **PyTorch Team** - Deep learning framework
- **Muay Thai Community** - Domain expertise and feedback

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the project maintainer.

---

**Built with ❤️ for the Muay Thai community**
