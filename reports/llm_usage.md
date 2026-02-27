#  LLM Usage in Gait Authentication System

This document explains where and how Large Language Models (LLMs) were used in this project.

---

## 🎯 Objective of Using LLM

The project requirement was to:

- Expand the dataset to 1000+ identities
- Improve model generalization
- Simulate realistic biomechanical diversity
- Strengthen training robustness

LLM was used as a synthetic profile generator to achieve this.

---

## 🧠 Where LLM Was Used

### 1️⃣ Biomechanical Profile Generation

File:
```
llm_engine/biomechanical_profile_generator.py
```

LLM was used to generate:

- Synthetic walking styles
- Step length variations
- Gait asymmetry factors
- Speed differences
- Sensor noise variations
- Natural human walking diversity

Each synthetic subject was assigned:
- Unique biomechanical parameters
- Realistic walking characteristics

This allowed expansion from:
- 30 real UCI subjects
to
- 1000+ synthetic identities

---

### 2️⃣ Synthetic Signal Simulation

File:
```
llm_engine/synthetic_signal_simulator.py
```

Using LLM-generated biomechanical parameters, we:

- Simulated accelerometer signals (x, y, z)
- Simulated gyroscope signals (x, y, z)
- Generated realistic 50Hz walking signals
- Created multiple windows per synthetic subject

Final result:
- 30,000+ synthetic gait windows
- 1000+ virtual identities

---

## 📊 Impact of LLM Augmentation

| Without LLM | With LLM |
|-------------|----------|
| 30 subjects | 1030+ subjects |
| Limited diversity | High biomechanical diversity |
| Weak generalization | Strong generalization |
| Lower robustness | Higher robustness |

Final Identification Accuracy:
**95.82%**

---

## 🔐 Why LLM Is Important Here

Traditional data augmentation:
- Adds noise
- Rotates signals
- Scales signals

LLM-based augmentation:
- Generates semantically meaningful human gait variations
- Simulates realistic biomechanical differences
- Mimics inter-person variability

This creates a richer embedding space.

---

## 🚀 Deployment Benefit

Because of LLM-generated diversity:

- Real-world employees can be registered without retraining
- Model generalizes to unseen walking styles
- Authentication remains stable

---

## 🏆 Conclusion

LLM was used as a:

- Biomechanical diversity generator
- Synthetic identity scaler
- Training robustness enhancer

It enabled expansion to 1000+ identities while maintaining realistic gait characteristics.