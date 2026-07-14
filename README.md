#  Latent Divergence Modeling in Paired Non-Stationary Time Series

A research-oriented machine learning framework for inferring, tracking, and forecasting latent divergence dynamics between correlated agents under sparse and noisy observations.

Built using probabilistic state-space modeling and neural sequence learning.

---

## Overview

Many real-world systems evolve together while remaining only partially observable.

Examples include:

- Financial assets
- Human behavioral dynamics
- Biological systems
- Sensor networks
- Multi-agent systems

This project proposes a hybrid latent-state framework capable of modeling hidden divergence and convergence patterns in paired non-stationary time series.

---

## Research Question

> How can we robustly infer, track, and predict latent divergence between correlated agents under sparse and noisy observations?

---

## Key Contributions

### Hybrid Latent State Model

- Neural sequence encoder
- Latent State Space Model (SSM)
- Variational latent inference

---

### Divergence-Aware Objective

Custom loss function

\[
L =
L_{KL}
+
\lambda L_{div}
+
\beta L_{smooth}
\]

where

- KL preserves latent structure
- Divergence penalizes persistent separation
- Smoothness encourages temporal consistency

---

### Regime-Aware Modeling

Supports multiple latent operating regimes.

Useful for

- structural breaks
- behavioral shifts
- non-stationary dynamics

---

## Repository Structure

```
latent-divergence-modeling/

│
├── analysis/
│   ├── visualize_latents.py
│   ├── divergence_metrics.py
│   ├── evaluate.py
│   └── ablations.py
│
├── baselines/
│   ├── lstm.py
│   ├── hmm.py
│   └── kalman.py
│
├── data/
│   ├── synthetic_generator.py
│   └── dataset.py
│
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── ablations.py
│
├── losses/
│   └── divergence_loss.py
│
├── models/
│   ├── encoder.py
│   ├── latent_ssm.py
│   └── regime_model.py
│
├── results/
│
├── app.py
├── run_train.py
├── requirements.txt
└── README.md
```

---

## Model Pipeline

```
Observed Time Series
        │
        ▼
Neural Encoder
        │
        ▼
Latent Representation
        │
        ▼
State Space Model
        │
        ▼
Latent Trajectories
        │
        ▼
Divergence Estimation
        │
        ▼
Visualization & Analysis
```

---

## Baselines

Implemented comparison models

- LSTM
- Hidden Markov Model (HMM)
- Kalman Filter

---

## Experiments

Current experiments include

- Synthetic paired time series
- Latent trajectory inference
- Divergence visualization
- Regime-aware analysis
- Baseline comparison
- Ablation studies

---

## Results

The framework visualizes

- Latent trajectories
- Divergence dynamics
- Regime transitions
- Mean divergence statistics

Example outputs are available in the `results/` directory.

---

## Installation

```bash
git clone https://github.com/gotnochill815-web/latent-divergence-modeling.git

cd latent-divergence-modeling

pip install -r requirements.txt
```

---

## Training

```bash
python run_train.py
```

---

## Visualization

```bash
python -m analysis.visualize_latents
```

---

## Streamlit Demo

Run locally

```bash
streamlit run app.py
```

---

## Future Work

- Transformer-based temporal encoder
- Variational State Space Models
- Bayesian uncertainty estimation
- Real-world financial datasets
- Multivariate latent dynamics
- Online inference
- Diffusion-based temporal forecasting

---

## Applications

- Quantitative Finance
- Time-Series Forecasting
- Multi-Agent Systems
- Behavioral Modeling
- Sensor Fusion
- Computational Neuroscience
- Healthcare Analytics

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Streamlit

---

## Citation

If you find this project useful, please consider citing the repository.

---

## Author

**Prakhya Khandelwal**

AI/ML Research | Probabilistic Modeling | Time-Series Learning | Generative AI
