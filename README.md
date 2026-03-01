# 🚶 Gait-Based Employee Authentication System

A deep learning–based biometric authentication system using accelerometer and gyroscope sensor data.

This system identifies employees based on their walking patterns using a **Triplet-Loss Embedding Network** and **Cosine Similarity–based verification**, enhanced with **LLM-driven synthetic dataset expansion**.

---

## 📌 Overview

This project implements a biometric access control system using:

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Deep Metric Learning (Triplet Loss)
- 128-Dimensional Gait Embeddings
- Cosine Similarity for Authentication
- LLM-Generated Biomechanical Profiles
- Physics-Based Signal Simulation

The system supports:

- Employee Registration
- Real-Time Authentication
- Global Channel-Wise Normalization
- Streamlit Web Deployment
- Scalable Dataset Expansion (1000+ Identities)

---

## 🧠 Model Architecture

The system uses a 1D Convolutional Neural Network trained with Triplet Loss to learn a discriminative embedding space.

**Architecture Details:**

- 1D CNN Feature Extractor
- Fully Connected Embedding Layer
- Embedding Dimension: 128
- Loss Function: Triplet Loss
- Similarity Metric: Cosine Similarity
- Threshold-Based Access Decision

The network ensures that walking samples from the same person are embedded closer together while different individuals are separated in embedding space.

---

## 📊 Training Dataset

| Source | Subjects |
|--------|----------|
| UCI HAR Dataset | 30 |
| LLM-Generated Synthetic Profiles | 1000+ |
| **Total Identities** | **1030+** |

Final Identification Accuracy: **95.82%**

---

## 📥 Dataset Information

### 1️⃣ UCI HAR Dataset

The UCI Human Activity Recognition dataset was used as the base real-world dataset.

Download from:

https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

⚠️ Due to size limitations, the raw dataset is not included in this repository.

After downloading, place the extracted dataset inside:

```
data/raw/uci/
```

The system uses:

- body_acc_x/y/z
- body_gyro_x/y/z
- WALKING activity only

---

### 2️⃣ Real-World Data Collection

Real walking data was collected using the Android application:

Physics Toolbox Sensor Suite:

https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite&hl=en&gl=U0

**Configuration Used:**

- Sampling Rate: 50 Hz
- Sensors:
  - Accelerometer (x, y, z)
  - Gyroscope (x, y, z)
- Recording Duration: 15–60 seconds

Each employee folder must contain:

```
EMP_001/
├── accelerometer.csv
└── gyroscope.csv
```

### Real Data Preprocessing

- Gravity offset removal
- Resampling to 50 Hz
- Sliding window segmentation (128 samples)
- Global channel-wise normalization (same as training)

---

## 🤖 LLM Usage

LLM was used to generate **correlated biomechanical walking profiles** grounded in UCI HAR population statistics.

### Step 1: Profile Generation

The LLM generates realistic biomechanical parameters such as:

- Cadence
- Vertical acceleration amplitude
- Gyroscope dynamics
- Step asymmetry
- Heel-strike sharpness

### Step 2: Physics-Based Signal Simulation

A custom simulator converts these profiles into:

- Multi-harmonic inertial signals
- Heel-strike transient modeling
- Left-right asymmetry
- 6-axis accelerometer + gyroscope time-series windows

Files:

```
llm_engine/biomechanical_profile_generator.py
llm_engine/synthetic_signal_simulator.py
```

This expands the dataset to 1000+ identities while maintaining biomechanical realism and improving embedding generalization.

---

## 🔄 System Workflow

### 1️⃣ Offline Training

Train the model using:

```
python -m models.training_pipeline
```

This process:

- Loads UCI + synthetic windows
- Computes global channel-wise mean & standard deviation
- Applies consistent normalization
- Trains the Triplet embedding model
- Saves:
  - gait_embedding_model.pth
  - normalization_params.npz

---

### 2️⃣ Employee Registration

- Upload accelerometer.csv
- Upload gyroscope.csv
- Preprocess and normalize data
- Generate gait embedding
- Store embedding as biometric fingerprint

---

### 3️⃣ Authentication

- Upload new walking sample
- Apply identical preprocessing
- Generate embedding
- Compare with registered employees

Authentication Logic:

```
if cosine_similarity >= threshold:
ACCESS GRANTED
else:
ACCESS DENIED
```

Default threshold: **0.65**

---

## 📂 Project Structure

```
gait_authentication_system/
│
├── app/ # Streamlit Web UI
├── authentication/ # Similarity & Access Logic
├── preprocessing/ # Data Cleaning & Windowing
├── models/ # Neural Network & Training
├── llm_engine/ # LLM-Based Synthetic Expansion
├── data/ # Raw & Processed Data (Not Included)
├── reports/ # Documentation
├── requirements.txt
└── README.md
```

---

## 🚀 Run the System

### Install Dependencies

```
pip install -r requirements.txt
```

### Train Model

```
python -m models.training_pipeline
```

### Launch Web App

```
streamlit run app/streamlit_app.py
```

---

## 🏆 Key Highlights

- 1000+ LLM-augmented synthetic identities
- Physics-based gait signal simulation
- Triplet-loss metric learning
- Global normalization consistency
- Accelerometer + Gyroscope sensor fusion
- Real-time biometric authentication
- Streamlit-based deployment
- Domain-generalized gait representation

---

## 📎 Technologies Used

- Python
- PyTorch
- Streamlit
- NumPy
- SciPy
- Pandas

---

## 👤 Author

Ravipal  
Integrated M.Tech CSE  
Data Science & Machine Learning  
VIT Bhopal University