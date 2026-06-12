# 🛡️ AI Surveillance System

## Overview
An AI-powered surveillance system that performs real-time object detection, tracking, intrusion monitoring, and event logging using YOLOv8, DeepSORT, and Streamlit.

## Features
- Real-time object detection using YOLOv8
- Multi-object tracking using DeepSORT
- Intrusion detection
- CSV event logging
- Interactive Streamlit dashboard
- Downloadable detection reports

## System Architecture

```text
Video/Webcam
      ↓
YOLOv8 Detection
      ↓
DeepSORT Tracking
      ↓
Intrusion Detection
      ↓
CSV Logging
      ↓
Streamlit Dashboard
```

## Tech Stack
- Python
- YOLOv8
- DeepSORT
- OpenCV
- Streamlit
- Pandas

## Run the Project

### Start Detection System

```bash
python main.py
```

### Start Dashboard

```bash
streamlit run dashboard.py
```