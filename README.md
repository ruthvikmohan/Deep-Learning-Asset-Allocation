# WQU MScFE 642 Deep Learning for Finance — Group Work Project 1

An enterprise-grade deep learning and tactical asset allocation framework engineered for the WorldQuant University (WQU) MScFE 642 curriculum. This pipeline designs, calibrates, and backtests progressive neural network architectures to capture non-linear, cross-asset short-term market trends across five foundational ETF macro asset classes (SPY, TLT, SHY, GLD, DBO).

---

## 🔬 Deep Learning Forecasting Architecture

The system features five sequential computational phases mapping raw time-series data to cross-sectional trade execution:

### 1. Data Processing & Stationary Validation (`Step 1`)
* **Multi-Asset Sync:** Automates daily data mining for the full ETF space over the training/testing horizon (Jan 1, 2018 – Dec 30, 2022).
* **Stationary Transformation:** Applies Augmented Dickey-Fuller (ADF) asymptotic testing arrays to justify nominal price vector transformations into risk-neutral daily log-returns.
* **Joint Profile Dynamics:** Tracks rolling empirical covariance matrices and cross-sectional asset return profiles.

### 2. Multi-Layer Perceptron (MLP) Baseline (`Step 2`)
* **Lag Space Engineering:** Builds standalone feedforward networks for each asset class, mapping high-dimensional lookback lag vectors to a target 25-day ahead return window.
* **Regression Target:** Implements parametric training check passes using Mean Squared Error (MSE) objective boundaries.

### 3. Spatial CNN-GAF Image Classification (`Step 3`)
* **Matrix Encoding:** Transforms 1D return series into 2D geometric textures via Gramian Angular Fields (GAF). It maps temporal vectors into polar coordinate matrices, preserving non-linear angular alignments

* **Computer Vision Processing:** Deploys Convolutional Neural Networks (CNN) with spatial max-pooling and feature extraction kernels to parse multi-period correlation shapes.

### 4. Recurrent Architectures via Single-Output LSTMs (`Step 4`)
* **Sequential Long-Term Memory:** Deploys Long Short-Term Memory (LSTM) cells equipped with input, forget, and output gating mechanisms to capture temporal dependencies and volatility persistence without exploding/vanishing gradients.

### 5. Multi-Output Multi-Task Learning Network (`Step 5`)
* **Unified Deep Architecture:** Blends the hidden state representations of the independent LSTM modules into a joint, multi-output deep neural network.
* **Cross-Asset Dependency Tracking:** Calibrates a centralized feature layer that processes all five asset streams simultaneously, outputting a 5-dimensional vector of 25-day ahead returns to capture hidden spillover effects.

### 6. Cross-Sectional Tactical Asset Allocation Backtester (`Step 6 & 7`)
* **Portfolio Optimization Engine:** Generates cross-sectional tactical asset allocation choices every 25 days based on out-of-sample multi-output predictions (e.g., long the top 2 predicted ETFs, short the bottom 2).
* **Performance Benchmark:** Backtests the multi-output network against a passive Buy-and-Hold Equally Weighted (1/5 each) portfolio baseline, reporting Cumulative Returns, Sharpe Ratios, and Max Drawdowns.

---

## 🛠️ Quantitative Tech Stack

* **Language Platform:** Python 3.10+
* **Deep Learning Engine:** `keras` / `tensorflow` (Functional API for multi-output graphs, Conv2D, LSTM)
* **Time-Series Image Conversions:** `pyts.image.GramianAngularField`
* **Data Retrieval & Verification:** `yfinance`, `statsmodels.tsa.stattools` (ADF test)
* **Matrix Calculus Engine:** `numpy`, `pandas`, `scipy`
* **Plotting & Architectural Visualizations:** `matplotlib`, `seaborn`

---

## ⚙️ Running the Pipeline Locally

1. Clone this repository:
   ```bash
   git clone https://github.com
   cd WQU-MScFE642-GWP1
   ```

2. Establish an isolated virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use: venv\Scripts\activate
   pip install numpy pandas matplotlib seaborn scipy yfinance statsmodels tensorflow pyts jupyter
   ```

3. Launch the environment:
   ```bash
   jupyter notebook
   ```

---

## ⚖️ Academic Integrity & AI Policy
This implementation adheres strictly to the **WorldQuant University Academic Policy Standards**. It is hosted openly purely as a peer-to-peer verification reference template for the MScFE curriculum. Hard copying or turning in sections of this framework directly will violate university honor codes and flag plagiarism filters in Turnitin.
