# app/streamlit_app.py
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import torch
import os
import tempfile
import numpy as np

from models.embedding_model import GaitEmbeddingModel
from preprocessing.real_data_loader import RealWorldDataLoader
from authentication.similarity_engine import SimilarityEngine


# -----------------------------
# LOAD MODEL (only once)
# -----------------------------
@st.cache_resource
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get absolute path to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "gait_embedding_model.pth")

    model = GaitEmbeddingModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# -----------------------------
# GENERATE EMBEDDING
# -----------------------------
def generate_embedding(model, device, windows):

    embeddings = []

    with torch.no_grad():
        for window in windows:
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            emb = model(tensor)
            embeddings.append(emb.squeeze(0).cpu().numpy())

    embeddings = np.array(embeddings)
    mean_embedding = np.mean(embeddings, axis=0)
    mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    return torch.tensor(mean_embedding, dtype=torch.float32)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Gait Authentication System", layout="centered")

st.title("🚶‍♂️ Gait-Based Employee Authentication System")

model, device = load_model()
engine = SimilarityEngine(threshold=0.65)
engine.load_registry()

menu = st.sidebar.selectbox(
    "Select Mode",
    ["Register Employee", "Authenticate Person", "System Info"]
)

# ============================
# REGISTER
# ============================
if menu == "Register Employee":

    st.header("🟢 Register New Employee")

    employee_id = st.text_input("Enter Employee ID (e.g., EMP_009)")

    accel_file = st.file_uploader("Upload Accelerometer CSV", type=["csv"])
    gyro_file = st.file_uploader("Upload Gyroscope CSV", type=["csv"])

    if st.button("Register"):

        if employee_id and accel_file and gyro_file:

            with tempfile.TemporaryDirectory() as tmpdir:

                person_dir = os.path.join(tmpdir, employee_id)
                os.makedirs(person_dir)

                accel_path = os.path.join(person_dir, "accelerometer.csv")
                gyro_path = os.path.join(person_dir, "gyroscope.csv")

                with open(accel_path, "wb") as f:
                    f.write(accel_file.read())

                with open(gyro_path, "wb") as f:
                    f.write(gyro_file.read())

                loader = RealWorldDataLoader(base_path=tmpdir)
                windows, labels = loader.load_all()

                embedding = generate_embedding(model, device, windows)

                engine.save_employee_embedding(employee_id, embedding)

                st.success(f"Employee {employee_id} Registered Successfully ✅")

        else:
            st.error("Please upload both files and enter Employee ID.")


# ============================
# AUTHENTICATE
# ============================
elif menu == "Authenticate Person":

    st.header("🔵 Authenticate Person")

    accel_file = st.file_uploader("Upload Accelerometer CSV", type=["csv"], key="auth_acc")
    gyro_file = st.file_uploader("Upload Gyroscope CSV", type=["csv"], key="auth_gyro")

    if st.button("Authenticate"):

        if accel_file and gyro_file:

            with tempfile.TemporaryDirectory() as tmpdir:

                person_dir = os.path.join(tmpdir, "UNKNOWN")
                os.makedirs(person_dir)

                accel_path = os.path.join(person_dir, "accelerometer.csv")
                gyro_path = os.path.join(person_dir, "gyroscope.csv")

                with open(accel_path, "wb") as f:
                    f.write(accel_file.read())

                with open(gyro_path, "wb") as f:
                    f.write(gyro_file.read())

                loader = RealWorldDataLoader(base_path=tmpdir)
                windows, labels = loader.load_all()

                embedding = generate_embedding(model, device, windows)

                match_id, score, status = engine.authenticate(embedding)

                st.subheader("Result")

                st.write("Best Match:", match_id)
                st.write("Similarity Score:", round(score, 4))

                if status == "ACCESS GRANTED":
                    st.success("ACCESS GRANTED ✅")
                else:
                    st.error("ACCESS DENIED ❌")

        else:
            st.error("Please upload both CSV files.")


# ============================
# SYSTEM INFO
# ============================
# ============================
# SYSTEM INFO
# ============================
else:

    st.header("🟣 System Information")

    st.subheader("📦 Model Details")

    st.write("Model Name: Triplet-Based Gait Embedding Network")
    st.write("Architecture: 1D CNN + Fully Connected Embedding Layer")
    st.write("Embedding Dimension: 128")
    st.write("Loss Function: Triplet Loss")
    st.write("Similarity Metric: Cosine Similarity")

    st.divider()

    st.subheader("📊 Training Data")

    st.write("UCI HAR Subjects: 30")
    st.write("LLM-Generated Synthetic Subjects: 1000+")
    st.write("Total Training Identities: 1030+")
    st.write("Identification Accuracy: 95.82%")

    st.divider()

    st.subheader("🔐 Deployment Settings")

    st.write("Registered Employees:", len(engine.registry))
    st.write("Similarity Threshold:", engine.threshold)
    st.write("Sensor Types: Accelerometer (x,y,z) + Gyroscope (x,y,z)")
    st.write("Window Size: 128 samples")
    st.write("Sampling Rate: 50 Hz")

    st.success("System Status: Operational ✅")