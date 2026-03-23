# SecureSpatial-FL: Production Implementation Instructions

## 1. Project Overview
Implement the "SecureSpatial-FL" framework, a privacy-preserving and decentralized system for WiFi-based human presence detection. The goal is to achieve 95%+ sensing accuracy in Non-Line-of-Sight (NLoS) environments.

## 2. Technical Pillars

### Phase 1: Signal Preprocessing (Perception)
- **CFR Modeling**: Implement the Channel Frequency Response (CFR) to isolate human gait signatures:
  $$H(f,t)=\sum_{k=1}^{K}a_{k}(t)e^{-j2\pi f\tau_{k}(t)}$$
- **Noise Reduction**: Apply low-pass filtering to raw CSI frames.
- **Feature Extraction**: Convert processed signals into 2D periodograms for spatial-temporal analysis.

### Phase 2: Hybrid AI Model (Trans-CRL)
Develop the "Trans-CRL" architecture using **PyTorch**:
- **Transformers**: Multi-head self-attention to capture signal fluctuations.
- **CNN**: Extraction of spatial features from CSI periodograms.
- **LSTM**: Modeling the temporal evolution of human movement.

### Phase 3: Decentralization (Federated Learning)
- **Local Training**: Edge nodes train locally on their own CSI signatures.
- **FedAvg Algorithm**: Implementation of Federated Averaging for weight aggregation:
  $$w_{t+1}=\sum_{k=1}^{K}\frac{n_k}{n}w_{k,t+1}$$
- **Privacy**: Raw data remains local; only model weights ($w$) are synchronized.

### Phase 4: Security (Blockchain Mutual Authentication)
- **Protocol**: Integrate a lightweight Blockchain-Based Remote Mutual Authentication (B-RMA) protocol.
- **Trust-Gate**: Ensure only authorized IoT nodes participate to prevent poisoning attacks.

## 3. Development Milestones & File Structure

1.  **datasets.py**: 
    - Automate downloading of datasets (e.g., Widar3.0, BGL) into a `/datasets` folder.
    - Implement data-loading classes for PyTorch `DataLoader` compatibility.
2.  **preprocessing.py**: 
    - Modular functions for CSI cleaning, filtering, and periodogram generation.
3.  **model.py**: 
    - PyTorch implementation of the Trans-CRL hybrid architecture.
4.  **blockchain_auth.py**: 
    - Node authentication logic and lightweight ledger simulation for the B-RMA protocol.
5.  **federated_logic.py**: 
    - Implementation of the FedAvg aggregation and local client update steps.
6.  **train.py**: 
    - The primary training script to orchestrate local and global training rounds.
    - **Logging**: Integrate `torch.utils.tensorboard` to save logs in a `/logs` folder for visualization.
7.  **visualize.py**: 
    - Scripts to generate plots for raw/processed CSI data and training results (accuracy/loss curves) from TensorBoard logs.

## 4. Environment & Tools
- **Framework**: PyTorch.
- **Logging**: TensorBoard (output to `./logs`).
- **Data Storage**: Local `./datasets` folder.