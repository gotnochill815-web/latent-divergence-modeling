#  Latent Divergence Modeling

> **Variational State Space Models for Learning Divergence Dynamics in Paired Time Series**

[![Demo](https://img.shields.io/badge/Live-Demo-red?style=for-the-badge)](https://latent-divergence-modeling-cznjz8yyqb49thkkqbpdta.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange?style=for-the-badge)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge)]()

---

## 🚀 Live Demo

**Try the interactive demo here**

https://latent-divergence-modeling-cznjz8yyqb49thkkqbpdta.streamlit.app/

---

# Motivation

Imagine monitoring two systems that should behave similarly.

Examples include

🏥 Two patients recovering from the same treatment

🏭 Two industrial machines operating under identical conditions

📈 Two correlated stock prices

🚗 Two autonomous vehicles following the same route

🤖 Two robots collaborating on a task

Initially both systems behave similarly.

Over time, subtle differences begin to emerge.

Traditional forecasting models mainly answer:

> **"What will happen next?"**

This project instead asks

> **"How are these two systems gradually becoming different?"**

The proposed Variational State Space Model learns this divergence directly in a latent space, allowing earlier detection of behavioral changes before they become obvious in the raw observations.

---

# Architecture

<p align="center">
<img src="docs/figures/architecture.png" width="900">
</p>

The model consists of

- Variational Encoder
- Latent State Space Model
- Shared Decoder
- Divergence-aware Latent Representation
- Multi-objective Training Objective

---

# Training Pipeline

<p align="center">
<img src="docs/figures/pipeline.png" width="900">
</p>

```
Synthetic Paired Time Series
            │
            ▼
    Variational Encoder
            │
            ▼
     Shared Latent Space
            │
            ▼
  Latent State Space Model
            │
            ▼
      Shared Decoder
            │
            ▼
 Reconstruction + Divergence Loss
            │
            ▼
         Evaluation
            │
            ▼
         Benchmarking
```

---

# Why Latent Divergence?

Instead of comparing observations directly,

```
Observation A
Observation B
```

the model learns

```
Latent Representation A

vs

Latent Representation B
```

This makes divergence estimation

- More robust to noise
- More interpretable
- Better suited for temporal dynamics
- Capable of identifying hidden behavioral changes

---

# Features

✅ Variational State Space Model

✅ Variational Encoder

✅ Shared Decoder

✅ Learnable Latent Dynamics

✅ KL Regularization

✅ Divergence-aware Learning

✅ Temporal Smoothness Constraint

✅ Synthetic Data Generator

✅ Kalman Filter Baseline

✅ Hidden Markov Model Baseline

✅ LSTM Baseline

✅ Automatic Benchmarking

✅ Interactive Streamlit Dashboard

---

# Repository Structure

```text
latent-divergence-modeling/

analysis/
baselines/
checkpoints/
data/
docs/
evaluation/
experiments/
losses/
models/
results/

app.py
README.md
requirements.txt
```

---

# Installation

```bash
git clone https://github.com/gotnochill815-web/latent-divergence-modeling.git

cd latent-divergence-modeling

pip install -r requirements.txt
```

---

# Training

```bash
python -m experiments.train
```

This

- trains the encoder
- trains the latent state-space model
- saves checkpoints
- records training history

---

# Evaluation

```bash
python -m experiments.evaluate
```

Outputs

- Mean Divergence
- Maximum Divergence
- Cosine Divergence
- MSE
- AUC

---

# Benchmark

```bash
python -m evaluation.benchmark
```

Benchmarks against

- Kalman Filter
- Hidden Markov Model
- LSTM
- Variational State Space Model (Ours)

---

# Training Dynamics

<p align="center">
<img src="docs/figures/training_curve.png" width="900">
</p>

The optimization process shows stable convergence with decreasing reconstruction error while simultaneously improving latent smoothness and divergence learning.

---

# Reconstruction

<p align="center">
<img src="docs/figures/reconstruction.png" width="900">
</p>

Comparison between the reconstructed paired observations and the original synthetic signals.

---

# Latent Space

<p align="center">
<img src="docs/figures/latent_space.png" width="750">
</p>

The encoder projects both time series into a shared latent space where divergence can be measured more effectively than in the raw observation space.

---

# Benchmark Comparison

<p align="center">
<img src="docs/figures/benchmark_summary.png" width="900">
</p>

Evaluation metrics include

- RMSE
- MAE
- Runtime
- Mean Divergence
- Cosine Divergence
- Smoothness

---

# Loss Function

The optimization objective combines

\[
\mathcal{L}
=
\mathcal{L}_{Recon}
+
\beta \mathcal{L}_{KL}
+
\lambda \mathcal{L}_{Div}
+
\gamma \mathcal{L}_{Smooth}
\]

where

- **Reconstruction Loss** preserves observations.
- **KL Divergence** regularizes the latent posterior.
- **Divergence Loss** separates paired latent trajectories.
- **Smoothness Loss** encourages temporally consistent latent dynamics.

---

# Potential Applications

🏥 Patient health monitoring

🏭 Predictive maintenance

🚗 Autonomous driving

📈 Financial market analysis

⚡ Industrial sensor monitoring

🛰 Satellite trajectory analysis

🤖 Multi-agent robotics

🌦 Climate and weather forecasting

---

# Technologies

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- Matplotlib
- FilterPy
- hmmlearn

---

# Future Work

- Real-world datasets
- Bayesian latent dynamics
- Transformer-based temporal modeling
- Diffusion priors
- Online inference
- Multi-modal sensor fusion
- Uncertainty-aware forecasting

---

# Author

**Prakhya Khandelwal**

AI Research • Machine Learning • Probabilistic Modeling • Deep Learning

GitHub

https://github.com/gotnochill815-web

---

# License

MIT License
