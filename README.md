📖 Overview
This project implements an end-to-end face mask detection pipeline using the YOLOv8 object detection architecture. It enables users to train a custom YOLOv8 model on face mask datasets, evaluate performance metrics, and deploy the trained model for real-time detection through a browser-based webcam interface — all within Google Colab.
The notebook is designed for accessibility and efficiency, leveraging GPU acceleration for training and inference while maintaining a clear, modular structure suitable for both educational purposes and practical deployment.

✨ Features
🎯 Core Capabilities

Custom Dataset Training — Train YOLOv8 models on your own face mask detection dataset
Multi-Class Detection — Supports detection of multiple mask-wearing states (with mask, without mask, incorrect mask, etc.)
GPU Acceleration — Optimized for NVIDIA T4 GPUs available in Google Colab
Automated Data Pipeline — Seamless dataset copying from Google Drive to local storage for faster training
Comprehensive Evaluation — Detailed validation metrics including mAP, precision, recall, and confusion matrices
Real-Time Webcam Inference — Live face mask detection through browser webcam integration
Interactive Visualization — Sample predictions with annotated bounding boxes and confidence scores
Model Export — Automatic saving of trained weights and configuration to Google Drive

🛠️ Technical Features

Flexible Model Selection — Choose from YOLOv8n (nano), YOLOv8s (small), or YOLOv8m (medium)
Advanced Augmentation — Built-in image augmentation including flipping, mosaic, and HSV adjustments
Early Stopping — Patience-based training termination to prevent overfitting
YAML Configuration — Structured dataset specification following YOLOv8 standards
Training Visualization — Automatic generation of loss curves, metrics plots, and confusion matrices
JavaScript-Python Bridge — Seamless webcam frame capture using browser APIs


📂 Project Structure
face-mask-detection-yolov8/
│
├── 📓 Face_Mask_Detection_yolo.ipynb    ← Main training & inference notebook
├── 📄 README.md                          ← Project documentation
│
├── 📁 Dataset Structure (in Google Drive)
│   ├── train/
│   │   ├── images/                       ← Training images
│   │   └── labels/                       ← YOLO format annotations (.txt)
│   ├── valid/
│   │   ├── images/                       ← Validation images
│   │   └── labels/                       ← YOLO format annotations (.txt)
│   ├── test/
│   │   ├── images/                       ← Test images
│   │   └── labels/                       ← YOLO format annotations (.txt)
│   └── classes.txt                       ← Class names (one per line)
│
└── 📁 Output (in Google Drive)
    └── trained_model/
        ├── best.pt                       ← Best model weights
        └── face_mask_data.yaml           ← Dataset configuration

🚀 Getting Started
Prerequisites

A Google account for accessing Google Colab
A custom face mask detection dataset in YOLO format
Google Drive for dataset storage and model output
A modern web browser with webcam access (for live inference)

Dataset Preparation
Your dataset must follow the YOLO annotation format:

Images — JPG, JPEG, PNG, or BMP format
Labels — Text files with the same name as images, containing:

   class_id center_x center_y width height
where all coordinates are normalized (0.0 to 1.0)

classes.txt — A text file listing class names (one per line), e.g.:

   with_mask
   without_mask
   mask_weared_incorrect

Directory Structure — Organize as shown in the Project Structure section above

Installation

Open Google Colab
Upload the Face_Mask_Detection_yolo.ipynb notebook
Navigate to Runtime → Change runtime type and select T4 GPU
Run the first cell to install all required dependencies


🎓 Workflow
The notebook is organized into 12 sequential cells, each performing a specific task:
CellPurposeDescription1Environment SetupVerifies GPU availability, installs Ultralytics YOLOv8, and checks dependencies2Drive Mount & Path VerificationMounts Google Drive and validates the presence of dataset folders3Dataset AnalysisReads class names and counts images/labels in each split (train/valid/test)4Local Data CopyCopies dataset from Google Drive to local Colab storage for faster I/O during training5YAML ConfigurationGenerates the YAML file required by YOLOv8 for dataset specification6Data VisualizationDisplays sample images with ground-truth bounding boxes for verification7Model TrainingTrains the YOLOv8 model with customizable hyperparameters8Evaluation & MetricsValidates the trained model and displays performance metrics with plots9Test Set InferenceRuns predictions on test images and visualizes results in a grid10Model ExportSaves the best model weights and YAML configuration back to Google Drive11Live Webcam SetupPrepares the trained model for real-time webcam inference12Single Frame CaptureCaptures a single frame from webcam and runs inference for testing

🏋️ Training
Model Selection
Choose your YOLOv8 variant based on your accuracy vs. speed requirements:
ModelParametersSpeedAccuracyRecommended Foryolov8n.pt3.2M⚡ Fastest⭐ GoodEdge devices, mobile deploymentyolov8s.pt11.2M🚀 Fast⭐⭐ BetterBalanced real-time applicationsyolov8m.pt25.9M🏃 Moderate⭐⭐⭐ HighHigh-accuracy requirements
Training Configuration
The training process is configured with the following default hyperparameters:
ParameterValueDescriptionepochs50Maximum number of training epochsimgsz640Input image size (640×640)batch16Batch size (reduce to 8 if out-of-memory errors occur)patience15Early stopping patience (stops if no improvement for 15 epochs)lr00.01Initial learning ratelrf0.001Final learning rateoptimizerAdamWOptimizer algorithm
Data Augmentation
The following augmentation techniques are applied during training:

Horizontal Flip — 50% probability
Mosaic Augmentation — 100% enabled (combines 4 images into one)
HSV Color Jittering — Hue (±1.5%), Saturation (±70%), Value (±40%)
Vertical Flip — Disabled (0%)


📊 Evaluation Metrics
After training, the model is evaluated using the following metrics:
Primary Metrics

mAP50 — Mean Average Precision at IoU threshold 0.5
mAP50-95 — Mean Average Precision averaged across IoU thresholds from 0.5 to 0.95
Precision — Ratio of true positives to all positive predictions
Recall — Ratio of true positives to all actual positive instances

Visualizations
The notebook automatically generates:

Training Curves — Loss and metric progression over epochs
Confusion Matrix — Per-class prediction accuracy
F1-Confidence Curve — Optimal confidence threshold identification
Precision-Recall Curve — Model performance trade-off visualization


🎥 Live Detection
Webcam Inference
The notebook includes a JavaScript-Python bridge that enables real-time face mask detection directly from your browser's webcam:
How It Works

JavaScript Webcam API — Captures video frames in the browser
Base64 Encoding — Converts frames to base64 strings for transmission
Python Processing — YOLOv8 performs inference on each frame
Annotated Display — Results are drawn with bounding boxes, labels, and confidence scores
Live Visualization — Annotated frames are displayed in real-time

Usage

Cell 11 sets up the webcam model
Cell 12 captures a single frame for testing
Extend the loop in Cell 12 to create continuous real-time detection

When prompted, allow camera access in your browser to enable frame capture.

🎨 Visualization Examples
The notebook provides multiple visualization types:
Ground Truth Annotation
Displays sample training images with their labeled bounding boxes to verify dataset quality.
Test Set Predictions
Shows a grid of test images with model predictions, including:

Bounding boxes color-coded by class
Class labels above each box
Confidence scores (0.0 to 1.0)

Training Metrics
Automatically generated plots include:

Loss curves (box loss, classification loss, DFL loss)
mAP progression over epochs
Precision and recall evolution


⚙️ Customization
Modifying Hyperparameters
To adjust training settings, edit Cell 7:
pythonmodel.train(
    epochs    = 100,        # ← Increase for more training
    batch     = 32,         # ← Increase if GPU memory allows
    lr0       = 0.005,      # ← Lower learning rate for fine-tuning
    patience  = 20,         # ← More patience for convergence
    # ... other parameters
)
Changing Model Architecture
Replace the model initialization in Cell 7:
pythonmodel = YOLO('yolov8m.pt')   # Use medium model for better accuracy
Adjusting Confidence Threshold
For inference, modify the confidence parameter:
pythonresults = model.predict(image_path, conf=0.5)  # Higher = stricter detections

🔧 Troubleshooting
Common Issues and Solutions
Out of Memory (OOM) Errors

Reduce batch size in Cell 7 from 16 to 8 or 4
Use a smaller model (yolov8n instead of yolov8s)
Decrease image size from 640 to 416

Dataset Not Found

Verify Google Drive is mounted correctly in Cell 2
Check that the BASE_DIR path matches your Drive folder structure
Ensure all splits (train/valid/test) contain both images and labels folders

Low Accuracy After Training

Increase number of epochs (50 → 100+)
Verify dataset quality and annotation accuracy
Add more training data if available
Try a larger model architecture (yolov8s → yolov8m)

Webcam Not Working

Ensure browser permissions allow camera access
Check that you're running in HTTPS (Colab default)
Try refreshing the page if the camera feed doesn't appear

Training Interrupted

Google Colab has a 12-hour runtime limit for free users
Consider upgrading to Colab Pro for longer sessions
Save checkpoints periodically to Google Drive


📈 Performance Tips
Training Speed Optimization

Copy dataset to local Colab storage (Cell 4 does this automatically)
Use smaller image sizes (e.g., 416 instead of 640) for faster iterations
Enable mixed precision training if supported by your GPU

Accuracy Improvement

Collect more diverse training data
Ensure balanced class distribution
Use test-time augmentation (TTA) during inference
Fine-tune on domain-specific data

Inference Speed

Use YOLOv8n for fastest real-time performance
Reduce input image size for webcam inference
Batch multiple frames together if processing recorded video


🛠️ Technology Stack
ComponentTechnologyObject DetectionUltralytics YOLOv8Deep Learning FrameworkPyTorch 2.0+Computer VisionOpenCVNumerical ComputingNumPyData VisualizationMatplotlibNotebook EnvironmentGoogle ColabConfigurationPyYAMLProgress TrackingtqdmWebcam IntegrationJavaScript (MediaDevices API)

📝 Dataset Format Reference
YOLO Annotation Format
Each label file should contain one line per object:
class_id x_center y_center width height
Where:

class_id: Integer index (0-indexed) matching position in classes.txt
x_center: Normalized x-coordinate of bounding box center (0.0 - 1.0)
y_center: Normalized y-coordinate of bounding box center (0.0 - 1.0)
width: Normalized width of bounding box (0.0 - 1.0)
height: Normalized height of bounding box (0.0 - 1.0)

Example
For an image of size 1920×1080 with a face at pixel coordinates (500, 300, 700, 500):
0 0.3125 0.3704 0.1042 0.1852
Calculation:

x_center = (500 + 700) / 2 / 1920 = 0.3125
y_center = (300 + 500) / 2 / 1080 = 0.3704
width = (700 - 500) / 1920 = 0.1042
height = (500 - 300) / 1080 = 0.1852


💾 Model Export and Deployment
Saving Trained Weights
Cell 10 automatically exports:

best.pt — Model weights with the highest validation mAP
face_mask_data.yaml — Dataset configuration for future inference

These files are saved to your Google Drive for persistent storage and can be downloaded for deployment in other environments.
Using Trained Model Elsewhere
To use your trained model outside of Colab:

Download best.pt from Google Drive
Install Ultralytics: pip install ultralytics
Run inference:

pythonfrom ultralytics import YOLO
model = YOLO('path/to/best.pt')
results = model.predict('image.jpg')

🎯 Use Cases
This face mask detection system can be applied to:

Public Health Monitoring — Automated compliance checking in public spaces
Access Control Systems — Entry verification for mask-wearing requirements
Retail Analytics — Customer safety compliance tracking
Educational Institutions — Campus safety monitoring
Transportation Hubs — Passenger screening in airports, stations, and terminals
Healthcare Facilities — Staff and visitor mask compliance verification
Smart City Infrastructure — Large-scale public safety monitoring


📄 License
This project is released under the MIT License. You are free to use, modify, and distribute this software for any purpose, including commercial applications.
The YOLOv8 model weights and Ultralytics library are subject to the AGPL-3.0 License.

🙏 Acknowledgments

Ultralytics for developing and maintaining the YOLOv8 framework
Google Colab for providing free GPU-accelerated cloud computing
PyTorch team for the deep learning framework
OpenCV community for computer vision tools
The open-source community for continuous contributions and improvements


🤝 Contributing
Contributions are welcome! If you'd like to improve this project:

Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Open a pull request

Please ensure your contributions maintain the professional quality and educational focus of this project.

📧 Support
For questions, issues, or feature requests:

Open an issue on the GitHub repository
Check existing issues for similar problems and solutions
Refer to the Ultralytics YOLOv8 documentation for model-specific queries
