# Traffic Sign Recognition using YOLOv8 (Classification)
This project implements a **Traffic Sign Recognition system** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and the **Ultralytics YOLOv8 model (classification mode)**.  
The aim is to build a computer vision model that can **recognize and classify different types of traffic signs** (e.g., Stop, Speed Limit, Yield).

We use YOLOv8 instead of a traditional CNN to leverage **pretrained weights, strong augmentations, and deployment-ready tools**, making this project both **research-friendly** and **industry-ready**.

## 🚦 Why Traffic Sign Recognition?
Traffic sign recognition is an essential part of **autonomous driving** and **driver assistance systems (ADAS)**.  
Correct classification ensures that vehicles can understand road rules and respond safely.


## ⚙️ Process Workflow

### 1️⃣ Dataset Preparation
- Data from **Kaggle (GTSRB)**.  
- Converted to YOLOv8 folder-per-class format (train/val/test).  
- Important: remove duplicate prefixes in CSV, rebuild test from `Test.csv`.

### 2️⃣ Training
- YOLOv8s-cls pretrained on ImageNet.  
- ~40 epochs, batch size 64, with augmentations (MixUp, CutMix).  
- Automatically saves `best.pt` and `last.pt`.

### 3️⃣ Validation
- Monitors accuracy on held-out validation split.  
- Reports top-1 and top-5 accuracy.  

### 4️⃣ Testing
- Final test set evaluation with **chunked + streamed predictions** (RAM-safe).  
- Fix applied for **class index mismatch** (alphabetical vs numeric).  

### 5️⃣ Visual Checks
- Random correct & misclassified predictions.  
- Confusion matrix and per-class accuracy charts.
  
## Why YOLOv8?
**Advantages:**
- Pretrained weights (ImageNet) → faster convergence.  
- Strong augmentations built-in.  
- Unified framework: detection, segmentation, classification.  
- Deployment ready (ONNX, TFLite, CoreML, TensorRT).  
- Lightweight and fast (`yolov8n/s-cls` for edge devices).  

**Disadvantages:**
- Heavier than simple CNNs for small datasets.  
- Large backbones may cause OOM on free GPUs.  
- Pure classification models (EfficientNet, ViT) can outperform on some tasks.  


## 📊 Visualizations

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Per-Class Accuracy
![Per Class Accuracy](images/per_class_accuracy.png)

### Random Predictions (Correct & Misclassified)
![Random Predictions](images/random_predictions.png)

## 📈 Results
- Validation accuracy: ~99–100%  
- Test accuracy (after label remapping): ~97–98%  
- Most classes >95% accuracy, with minor drops in visually similar signs (e.g., speed limits).


