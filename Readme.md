# Book Attention Monitoring System

A real-time attention monitoring system that uses computer vision and deep learning to analyze user attention levels through webcam feed.

## Features

- Real-time attention monitoring using webcam
- Face detection and pose analysis using YOLO
- Gaze direction analysis using L2CS-Net
- Attention metrics calculation and session reporting
- Configurable monitoring duration and check intervals
- Detailed session logs with timestamps
- Support for both webcam and video file input

## Prerequisites

- Python 3.8+
- Webcam or video file
- Required Python packages (see `requirements.txt`)
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Jayant71/Book-Reading_Attentiton-Monitor.git
cd book-attention-monitor
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download required model files:

- Place `L2CSNet_gaze360.pkl` in the project root
- Place `<your-yolo-model>.onnx/pt` in the project root

## Project Structure

```bash
book-attention-monitor/
├── src/
│   ├── analysis/
│   │   ├── attention_analyzer.py
│   │   └── attention_monitor.py
│   ├── camera/
│   │   └── camera_manager.py
│   ├── session/
│   │   └── session_manager.py
│   ├── models/
│   │   ├── gaze_estimator.py
│   │   └── object_detector.py
│   ├── utils/
│   │   └── helpers.py
├── main.py
├── sample.env
├── requirements.txt
└── README.md
```

## Usage

Run the main script to start monitoring:

```bash
python main.py
```

The system will:

1. Initialize the webcam or video source
2. Start monitoring attention levels using YOLO for face detection and L2CS-Net for gaze analysis
3. Display real-time status
4. Generate a session report upon completion

### Input Sources

The system supports multiple input sources:

- Webcam (default)
- Video file
- IP camera stream

To change the input source, modify the `source` variable in `main.py`:

```python
# For webcam
source = 0

# For video file
source = "path/to/video.mp4"

# For IP camera
source = "http://ip:port/video"
```

## Configuration

The system uses the following models:

- YOLO for book detection
- L2CS-Net for gaze direction analysis

Model files required:

- `L2CSNet_gaze360.pkl`: Gaze analysis model
- `<your-yolo-model>.onnx/pt`: Book detection model
