# Demo Files

This folder contains demonstration materials for the RTM-FASS project.

## What to Include Here

### 1. Demo GIF (Required)
- **File**: `demo.gif` or `rtm_fass_demo.gif`
- **Size**: Keep under 10MB
- **Length**: 10-15 seconds
- **Content**: Show the system analyzing a fight with:
  - Real-time pose detection
  - Strike detection indicators
  - Live scoring overlay
  - Final scorecards

**How to create:**
```bash
# Convert video to GIF using ffmpeg
ffmpeg -i outputs/analyzed_test_video.mp4 -t 15 -vf "fps=10,scale=720:-1:flags=lanczos" demo/demo.gif
```

### 2. Summary Images (Recommended)
- `summary_example.jpg` - Example fight summary/scorecard
- `training_curves.png` - Model training progress (already in models/)
- `system_architecture.png` - System diagram (optional)

### 3. YouTube Video Link
Upload your best demo video to YouTube and link it in the main README.

## Current Files in Demo

Place your demo files here before pushing to GitHub.

## Usage in README

Add this to your main README.md:

```markdown
## 🎬 Demo

![RTM-FASS Demo](demo/demo.gif)

**Watch the full demo:** [YouTube Video Link](https://youtube.com/...)

### Example Outputs

![Fight Summary](demo/summary_example.jpg)
```
