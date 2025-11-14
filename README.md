📘 Real-Time ASL Static Gesture Recognition Using MediaPipe & Lightweight ML Models

This project implements a real-time American Sign Language (ASL) static gesture recognition system using MediaPipe Hands for landmark extraction and lightweight machine learning models (SVM & MLP) for classification.
The system runs fully on CPU, works at 55–60 FPS, and requires no GPU, making it ideal for laptops and edge devices.

🚀 Features

Real-time ASL gesture recognition using a standard 720p webcam

MediaPipe 21-point hand landmark extraction

Lightweight ML models (SVM, MLP) for fast inference

Custom dataset (self-recorded) of 25 static gestures

Landmark preprocessing pipeline

Wrist-relative normalization

Scale normalization

Coordinate flattening

Angle and vector features

Temporal smoothing & majority voting for stable outputs

Privacy-friendly (stores only landmark coordinates, not images/videos)

Modular architecture (easily extendable to dynamic gestures)

📂 Project Structure
├── data/
│   ├── landmarks/          # Saved .csv landmark data
│   ├── models/             # Trained SVM/MLP models
│   └── dataset_info.txt
├── src/
│   ├── collect_data.py     # Record gesture samples with MediaPipe
│   ├── train_model.py      # Train SVM/MLP on collected data
│   ├── preprocess.py       # Feature extraction + normalization
│   ├── realtime_predict.py # Real-time gesture detection pipeline
│   └── utils.py            # Helper functions
├── results/
│   ├── confusion_matrix.png
│   └── sample_outputs/
├── README.md
└── requirements.txt

🛠️ Technologies Used

Python 3.12.6

MediaPipe Hands

OpenCV

scikit-learn (SVM, MLP)

NumPy / Pandas / Matplotlib

📸 How It Works
1. Data Collection

You collect gesture samples using:

python src/collect_data.py


Each frame is converted into 21 hand landmarks × (x,y,z)
→ Stored as (21 × 3 = 63 features) per frame.

2. Preprocessing & Feature Engineering

Applied during training and real-time inference:

Convert landmarks to wrist-relative coordinates

Normalize scale (remove hand size differences)

Flatten features

Compute inter-finger angles & directional vectors

Standardize features

This produces 83–90D optimized feature vector.

3. Model Training

Train SVM and MLP using:

python src/train_model.py


Uses:

80/20 train-test split

5-fold cross validation

Model comparison

Saves best model to /data/models/

4. Real-Time Prediction

Run the live ASL translator:

python src/realtime_predict.py


Pipeline:

Webcam frame

MediaPipe → extract 21 landmarks

Preprocess features

Predict using SVM/MLP

Temporal voting

Display gesture text on screen

📊 Performance
Classification Accuracy (Static Gestures)
Model	Accuracy
SVM (RBF)	84.1%
MLP (3-layer)	82.7%
Runtime Performance

55–60 FPS on Windows 11 laptop

12–18ms inference time

Low CPU usage (~25–30%)

🔍 Confusion Matrix Sample (6 Extracted Gestures)

Gestures included:

Yes, No, Thank You, Hello, Please, I

Used to identify hard classes & improve dataset balance.

🔐 Ethics, Privacy & Safety

Only self-recorded samples were used

No public dataset or sensitive data was imported

Only landmark coordinates were stored

No faces, environments, or raw video frames saved

Entire processing done on-device (no cloud upload)

🌟 Project Novelty

This project is unique because it:

Uses fully custom dataset, not downloaded data

Performs optimized landmark preprocessing

Achieves deep-learning-like performance without GPUs

Runs consistently at real-time frame rates

Has temporal stability smoothing for non-flickering output

Ensures privacy-by-design by storing abstract landmark coordinates

Features confusion-guided dataset enhancement

Is modular and extendable to dynamic gestures or mobile apps

💡 Future Enhancements

Add LSTM/Transformer for dynamic gestures

Implement two-hand gesture recognition

Support continuous sentence translation

Deploy on Android/iOS using TensorFlow Lite

Add built-in ASL learning assistant mode

Build a web-based interface

📥 Installation

Install dependencies:

pip install -r requirements.txt

▶️ Run Real-Time Detection
python src/realtime_predict.py

📜 License

MIT License – free for academic and personal use.

👥 Contributors

Anil Kumar (24071A7226)

Nikhil (24071A7209)

Ananya Reddy (24071A7206)

Surya Arjun (24071A7212)

Department of CSE (CyS, DS) and AI&DS
VNR Vignana Jyothi Institute of Engineering & Technology
