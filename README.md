# 🛣️ Lane Line Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)

A robust **computer vision system** for real-time lane detection in video streams.
Built with **OpenCV** and optimized for **accuracy and stability** across diverse road conditions.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/santosh-dasu/lane-line-detection-system.git
cd lane-line-detection-system
pip install -r requirements.txt
```

### 2. Usage

#### Process videos

```bash
python process_video.py
```

#### Process challenge video (curves, shadows, harder cases)

```bash
python process_challenge.py
```

#### Process single images

```bash
python process_image.py
```

#### Unified pipeline with custom parameters

```bash
python unified_lane_detection.py input_video.mp4 -m auto -o output.mp4
```

* **Modes:**

  * `auto` – Automatically adjust parameters
  * `challenging` – For shadows, curves, complex roads
  * `easy` – For highways and straight lanes

---

## ✨ Features

* **Stable Detection** → No flickering, smooth lane tracking
* **Multiple Modes** → Auto, challenging, and easy
* **Color Processing** → HLS color space + Sobel edge detection
* **Temporal Smoothing** → Uses 5-frame history for stable detection
* **Error Handling** → Graceful fallbacks for failed detections

---

## 📁 Project Structure

```
├── unified_lane_detection.py    # Main detection system
├── process_challenge.py         # Quick challenge video processing
├── process_video.py             # Regular video processing
├── process_image.py             # Single image processing
├── requirements.txt             # Dependencies
├── test_image/                  # Sample images
├── test_vedios/                 # Sample videos
└── outputs/                     # Processed results
```

---

## 🔧 Requirements

* **Python**: 3.8+
* **OpenCV**: 4.8+
* **NumPy**: 1.24+
* **MoviePy**: 1.0.3+

---

> 🛠️ *Lane Detection System for Autonomous Vehicle Applications*

---
