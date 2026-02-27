# 🚶 Gait-Based Employee Authentication System

A deep learning–based biometric authentication system using accelerometer and gyroscope sensor data.

This system identifies employees based on their walking patterns using a triplet-loss embedding network and cosine similarity–based verification.

---

## 📌 Overview

This project implements a biometric access control system using:

- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Deep Metric Learning (Triplet Loss)
- 128-Dimensional Gait Embeddings
- Cosine Similarity for Authentication

The system supports:

- Employee Registration
- Real-Time Authentication
- Streamlit Web Deployment
- LLM-Based Synthetic Dataset Expansion

---

## 🧠 Model Architecture

- 1D Convolutional Neural Network
- Embedding Dimension: 128
- Loss Function: Triplet Loss
- Similarity Metric: Cosine Similarity
- Threshold-Based Access Decision

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

---

### 2️⃣ Real-World Data Collection

Real walking data was collected using the Android application:

Physics Toolbox Sensor Suite:

https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite&hl=en&gl=U0

Configuration used:

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

---

## 🤖 LLM Usage

LLM was used to:

- Generate biomechanical walking profiles
- Simulate realistic gait diversity
- Expand dataset to 1000+ identities
- Improve embedding generalization

Files:
```
llm_engine/biomechanical_profile_generator.py
llm_engine/synthetic_signal_simulator.py
```

This significantly improved model robustness.

---

## 🔄 System Workflow

### 1️⃣ Offline Training
```
python -m models.training_pipeline
```

- Train embedding model on UCI + synthetic data
- Save trained model weights

---

### 2️⃣ Employee Registration

- Upload accelerometer.csv
- Upload gyroscope.csv
- Generate gait embedding
- Store biometric fingerprint

---

### 3️⃣ Authentication

- Upload new walking sample
- Generate embedding
- Compare with registered employees
- Output:

```
ACCESS GRANTED
or
ACCESS DENIED
```

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

## 🔐 Authentication Logic

```
If cosine_similarity >= threshold:
ACCESS GRANTED
else:
ACCESS DENIED
```

Default threshold: 0.75

---

## 🏆 Key Highlights

- 1000+ LLM-augmented synthetic identities
- Triplet-loss embedding model
- Real-time biometric authentication
- Accelerometer + Gyroscope fusion
- Web deployment via Streamlit
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