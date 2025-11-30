# RTM-FASS
# 🥊 RTM-FASS: Real-Time Muay Thai Fight Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/RTM-FASS/blob/main/notebooks/demo.ipynb)

----
> An AI-powered automated scoring system for Muay Thai fights using computer vision and deep learning to provide objective scoring for each fighter

This was created for my Final Year Project at the University of Westminster.

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

---

## ✨ Key Features

### 🎥 Video Processing
- **Live & Recorded Analysis** - Process fights in real-time or from video files
- **Multi-Fighter Tracking** - Simultaneous detection and tracking of both fighters
- **Environment Adaptation** - Robust to varying lighting, angles, and camera distances
- **YouTube Integration** - Direct analysis from online fight videos

### 🦴 Computer Vision Pipeline
- **Fighter Detection** - RTMDet-M for accurate person detection
- **Pose Estimation** - YoloPosev8 with 17-point skeleton tracking
- **Automatic ID Assignment** - Left/right fighter identification based on ring position
- **Stance Recognition** - Orthodox vs Southpaw detection

### 👊 Strike Recognition System

Comprehensive detection of all Muay Thai techniques:

| Category | Techniques Detected |
|----------|-------------------|
| **Punches** | Jab, Cross, Hook, Uppercut |
| **Kicks** | Roundhouse, Front Kick, Side Kick, Low Kick |
| **Elbows** | Horizontal, Diagonal, Uppercut, Spinning |
| **Knees** | Straight, Curved, Flying, Jumping |
| **Clinch** | Position control, knee strikes in clinch |

### 📊 Scoring Engine

**Traditional 10-Point Must System**:
- Strike effectiveness weighted by technique type
- Aggression and ring control metrics
- Defense rate (blocks + evasions)
- Technical execution scoring
