# RTM-FASS Data Structure

This directory contains the training and evaluation datasets for the RTM-FASS project.

## Directory Structure

```
data/
├── raw_videos/              # Original fight footage (NOT included in repo)
│   ├── strikes/            # Padwork demonstration videos
│   └── fights/             # Full fight videos
├── annotations/            # COCO format annotations (NOT included in repo)
├── strike_dataset/         # Processed strike sequences (NOT included in repo)
├── fight_dataset/          # Processed fight sequences (NOT included in repo)
├── processed_data/         # Intermediate processing outputs (NOT included in repo)
└── *.csv                   # Metadata and labels (included in repo)
```

## What's Tracked in Git

✅ **CSV files** - Dataset metadata and labels
✅ **JSON metadata** - Configuration and class definitions (small files only)

## What's NOT Tracked (See .gitignore)

❌ **Video files** (.mp4, .avi, .mov) - Too large for Git
❌ **NumPy arrays** (.npy) - Processed sequences
❌ **Large annotations** - COCO JSON files
❌ **Excel files** (.xlsx) - Use CSV instead

## Dataset Statistics

- **Padwork Sequences**: Thousands of labeled technique demonstrations
- **Fight Sequences**: Real sparring footage with COCO annotations
- **Total Size**: ~9.7GB (excluded from repository)

## Getting the Data

Due to size constraints, the full dataset is not included in this repository. To obtain the data:

1. Contact the project maintainer
2. Use your own Muay Thai footage
3. Follow the data processing pipeline in `Copy of MAIN.ipynb`

## Creating Your Own Dataset

See the notebook `Copy of MAIN.ipynb` for complete data processing pipelines:
- Process padwork demonstrations
- Annotate sparring footage with COCO format
- Generate training sequences
