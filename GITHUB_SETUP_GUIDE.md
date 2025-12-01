# GitHub Repository Configuration Guide
## RTM-FASS Project

---

## 📊 Project Overview

**Total Size**: 10.74 GB (26,960 files)
- **Data folder**: 9.69 GB (26,924 files)
- **Models folder**: 435.89 MB (7 files)
- **Source code**: ~100 MB

---

## ✅ FILES TO INCLUDE IN GITHUB

### 📁 Root Level
- ✅ `README.md` - Main project documentation (CREATED)
- ✅ `LICENSE` - MIT License (CREATED)
- ✅ `requirements.txt` - Python dependencies (CREATED)
- ✅ `.gitignore` - Ignore rules (CREATED)
- ✅ `Copy of MAIN.ipynb` - Complete pipeline notebook

### 📁 Source Code (`src/`)
- ✅ `__init__.py`
- ✅ `config.py`
- ✅ `pose_detector.py`
- ✅ `strike_detector.py`
- ✅ `strike_model.py`
- ✅ `fight_analyzer.py`
- ✅ `visualization.py`
- ✅ `data_processor.py`
- ✅ `train_model.py`

### 📁 Data (`data/`)
- ✅ `README.md` - Data documentation (CREATED)
- ✅ `techniques_dataset.csv` - Main dataset labels
- ✅ `processed_techniques_dataset.csv` - Processed labels
- ✅ `processed_techniques_dataset_final.csv`
- ✅ `PXL_*.csv` - Video clip metadata
- ✅ `PXL_*.json` - Small JSON metadata files

### 📁 Models (`models/`)
- ✅ `README.md` - Model documentation (CREATED)
- ✅ `training_curves.png` - Training visualizations
- ✅ `combined_training_curves.png`

### 📁 Outputs (`outputs/`)
- ✅ Keep folder structure (add empty `.gitkeep`)
- ❌ Don't track actual output files

---

## ❌ FILES TO EXCLUDE FROM GITHUB

### 🚫 Large Binary Files
```
❌ *.pth (PyTorch models) - 145MB each
❌ *.pt (YOLO models) - yolov8n.pt, yolov8x-pose.pt
❌ *.npy (NumPy arrays) - Thousands of files
❌ *.mp4 (Videos) - All video files
❌ *.MP4, *.avi, *.mov
```

### 🚫 Data Files (~9.7GB)
```
❌ data/raw_videos/**/* - Original footage
❌ data/strike_dataset/*.npy - Processed sequences
❌ data/fight_dataset/*.npy - Fight sequences
❌ data/processed_data/**/* - Intermediate files
❌ data/annotations/*.json - Large COCO files
❌ data/pad_detection_dataset/
❌ data/pad_detector/
```

### 🚫 Generated/Temporary Files
```
❌ __pycache__/ - Python bytecode
❌ *.pyc, *.pyo, *.pyd
❌ .ipynb_checkpoints/ - Jupyter checkpoints
❌ outputs/*.mp4 - Analysis results
❌ outputs/*.jpg
❌ outputs/*.json
❌ *.xlsx - Excel files (use CSV)
```

### 🚫 Environment/IDE Files
```
❌ .claude/ - Claude AI settings
❌ .vscode/ - VS Code settings
❌ .idea/ - PyCharm settings
❌ venv/, env/ - Virtual environments
❌ .DS_Store - macOS files
❌ Thumbs.db - Windows files
```

---

## 📋 COMPLETE FILE CHECKLIST

### Root Directory Structure
```
RTM-FASS/
├── ✅ .gitignore
├── ✅ LICENSE
├── ✅ README.md
├── ✅ requirements.txt
├── ✅ Copy of MAIN.ipynb
├── ❌ yolov8n.pt (98MB)
├── ❌ yolov8x-pose.pt (149MB)
│
├── src/
│   ├── ✅ __init__.py
│   ├── ✅ config.py
│   ├── ✅ data_processor.py
│   ├── ✅ fight_analyzer.py
│   ├── ✅ pose_detector.py
│   ├── ✅ strike_detector.py
│   ├── ✅ strike_model.py
│   ├── ✅ train_model.py
│   ├── ✅ visualization.py
│   └── ❌ __pycache__/
│
├── data/
│   ├── ✅ README.md
│   ├── ✅ techniques_dataset.csv
│   ├── ✅ processed_techniques_dataset*.csv
│   ├── ✅ PXL_*.csv
│   ├── ✅ PXL_*.json (small files)
│   ├── ❌ *.xlsx
│   ├── ❌ raw_videos/ (9GB+)
│   ├── ❌ strike_dataset/ (thousands of .npy)
│   ├── ❌ fight_dataset/ (thousands of .npy)
│   ├── ❌ processed_data/
│   ├── ❌ annotations/
│   ├── ❌ pad_detection_dataset/
│   └── ❌ pad_detector/
│
├── models/
│   ├── ✅ README.md
│   ├── ✅ training_curves.png
│   ├── ✅ combined_training_curves.png
│   ├── ❌ best_model.pth (145MB)
│   ├── ❌ combined_strike_model.pth (145MB)
│   ├── ❌ FINAL_MODEL.pth (145MB)
│   ├── ❌ strike_model.pth
│   └── ❌ ultra_fast_model.pth
│
├── outputs/
│   ├── ✅ .gitkeep (create empty file)
│   ├── ❌ *.mp4
│   ├── ❌ *.jpg
│   └── ❌ *.json
│
└── training/
    ├── ✅ fighter_dataset/ (if small configs)
    └── ✅ strike_dataset/ (if small configs)
```

---

## 🔧 SETUP INSTRUCTIONS

### 1. Create `.gitkeep` for Empty Folders
```powershell
New-Item -Path "c:\Users\ylop\Downloads\RTM FASS\RTM-FASS\outputs\.gitkeep" -ItemType File
```

### 2. Initialize Git Repository
```bash
cd "c:\Users\ylop\Downloads\RTM FASS\RTM-FASS"
git init
git add .
git commit -m "Initial commit: RTM-FASS Fight Analysis System"
```

### 3. Verify What Will Be Committed
```bash
# Check what's being tracked
git status

# See ignored files
git status --ignored

# Check total size
git count-objects -vH
```

### 4. Create GitHub Repository
```bash
# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/RTM-FASS.git
git branch -M main
git push -u origin main
```

---

## 📦 HANDLING LARGE FILES

### Option 1: GitHub Releases (RECOMMENDED)
Upload large model files as release assets:
1. Go to GitHub → Releases → Create new release
2. Upload `.pth` files as binary attachments
3. Users download separately from releases

### Option 2: Git LFS (Large File Storage)
```bash
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Configure Git LFS"
```
**Note**: Git LFS has storage/bandwidth limits on free tier

### Option 3: External Storage
- Google Drive / Dropbox links in README
- Hugging Face Model Hub
- Cloud storage (AWS S3, Azure Blob)

---

## 📐 SIZE BREAKDOWN

### What WILL be committed (~100-200MB):
- Source code: ~5MB
- CSV metadata: ~10MB
- Notebook: ~2MB
- Documentation: ~1MB
- Training curves: ~2MB
- Small JSON files: ~5MB

### What WON'T be committed (~10.5GB):
- Models: 435MB (7 files)
- Raw videos: ~4GB
- Processed sequences (.npy): ~5GB
- Processed videos: ~1GB
- Annotations: ~500MB

---

## ⚠️ IMPORTANT WARNINGS

### Before First Commit:
1. ✅ Verify `.gitignore` is in place
2. ✅ Test with `git status --ignored`
3. ✅ Check repo size: `git count-objects -vH`
4. ✅ DO NOT commit if size > 500MB

### GitHub Limits:
- 📦 File size: 100MB max (enforced)
- 📦 Repository: 1GB recommended, 5GB warning
- 📦 Push size: 2GB max per push
- 🚨 Files > 100MB will **reject** your push

### Clean Up If Needed:
```bash
# If you accidentally commit large files:
git rm --cached models/*.pth
git rm --cached *.pt
git commit -m "Remove large model files"

# To completely clean history:
# Use BFG Repo-Cleaner or git-filter-branch
```

---

## 🎯 RECOMMENDED WORKFLOW

### Phase 1: Initial Setup ✅
1. ✅ `.gitignore` created
2. ✅ `README.md` created
3. ✅ `LICENSE` created
4. ✅ `requirements.txt` created
5. ✅ Documentation created

### Phase 2: First Commit
```bash
git init
git add .
git status  # Verify only small files
git commit -m "Initial commit"
```

### Phase 3: Push to GitHub
```bash
# Create repo on GitHub first
git remote add origin [URL]
git push -u origin main
```

### Phase 4: Handle Large Files
```bash
# Upload to GitHub Releases
# Add download links to README
# Update models/README.md with instructions
```

---

## 📝 FINAL CHECKLIST

- [x] `.gitignore` configured
- [x] `README.md` comprehensive
- [x] `LICENSE` included
- [x] `requirements.txt` complete
- [x] Documentation for data/models
- [ ] Test git status (verify <500MB)
- [ ] Create GitHub repo
- [ ] Initial commit & push
- [ ] Upload models to Releases
- [ ] Update README with download links
- [ ] Add repo badges/shields
- [ ] Create CONTRIBUTING.md (optional)
- [ ] Add example images/demo (optional)

---

## 🎓 REPOSITORY BEST PRACTICES

### Good README Features:
- ✅ Badges for Python, PyTorch, License
- ✅ Clear problem statement
- ✅ Architecture diagrams
- ✅ Installation instructions
- ✅ Usage examples
- ✅ Model performance metrics
- ✅ Citation/acknowledgments

### Repository Organization:
- ✅ Logical folder structure
- ✅ Clear separation of concerns
- ✅ Documentation in each folder
- ✅ Examples and demos
- ✅ Issue templates (optional)
- ✅ Contributing guidelines (optional)

---

## 📞 QUESTIONS?

If you encounter issues:
1. Check file sizes: `git ls-files -z | xargs -0 du -h | sort -hr | head -20`
2. Verify ignored: `git status --ignored`
3. Clean cache: `git rm -r --cached .`
4. Re-add: `git add .`

**Your repo is ready to push!** Total committed size will be ~100-200MB (well within limits).
