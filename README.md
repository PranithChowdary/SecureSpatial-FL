# SecureSpatial-FL

SecureSpatial-FL is a privacy-preserving and decentralized framework for WiFi-based human presence detection. It leverages Channel State Information (CSI) from WiFi signals to achieve high-accuracy sensing (95%+) even in Non-Line-of-Sight (NLoS) environments, while ensuring data privacy through Federated Learning and security through Blockchain-based authentication.

## 🚀 Key Features

*   **Signal Preprocessing (Perception):** Advanced CSI cleaning using Low-Pass Filtering (LPF) and Channel Frequency Response (CFR) modeling to isolate human gait signatures.
*   **Hybrid AI Model (Trans-CRL):** A state-of-the-art architecture combining:
    *   **CNNs:** Spatial feature extraction from 2D periodograms.
    *   **Transformers:** Multi-head self-attention to capture signal fluctuations.
    *   **LSTMs:** Modeling the temporal evolution of human movement.
*   **Decentralized Training (Federated Learning):** Privacy-preserving training using the **FedAvg** algorithm. Raw CSI data remains local on edge nodes; only model weights are synchronized.
*   **Secure Authentication (B-RMA):** Integrated Blockchain-Based Remote Mutual Authentication protocol to prevent poisoning attacks and unauthorized node participation.

## 📁 Project Structure

```text
SecureSpatial-FL/
├── datasets.py         # CSI Data loading and synthetic data generation
├── preprocessing.py    # Signal cleaning, LPF, and periodogram extraction
├── model.py            # Trans-CRL (CNN + Transformer + LSTM) implementation
├── federated_logic.py  # FedAvg aggregation and local update logic
├── blockchain_auth.py  # B-RMA node authentication simulation
├── train.py            # Main federated training orchestration script
├── visualize.py        # Utilities for signal and periodogram visualization
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SecureSpatial-FL
    ```

2. **Create virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Recommended Python version is 3.9+.*

## 💻 Usage

### 1. Run Federated Training
To start a simulated federated training session with multiple clients and blockchain authentication:
```bash
python3 train.py
```
This will:
*   Register and authenticate 3 simulated clients.
*   Generate synthetic CSI data for local training.
*   Run 5 global rounds of Federated Averaging.
*   Save the final global model to `./models/global_model.pth`.

### 2. Visualize CSI Processing
To see how signals are filtered and converted into periodograms:
```bash
python3 visualize.py
```
Outputs (plots) will be saved in the `./logs/` directory.

### 3. Individual Module Testing
You can test the preprocessing or model architecture independently:
```bash
python3 preprocessing.py  # Test signal processing logic
python3 model.py          # Verify Trans-CRL forward pass
```

## 🧠 Architecture Details

### Trans-CRL Model
The core of SecureSpatial-FL is the Trans-CRL model. It processes 2D periodograms (Frequency vs. Time) as input.
1.  **CNN Layers** reduce spatial dimensionality and extract local patterns.
2.  **Transformer Layers** treat spatial features as a sequence to focus on relevant signal fluctuations.
3.  **LSTM Layers** capture the long-term temporal dependencies of human movement (e.g., walking, sitting).

### Federated Learning & Security
*   **Privacy:** Local nodes train on their private datasets. Only updated weights are sent to the central server.
*   **Aggregation:** The central server uses `w_{t+1} = Σ (n_k/n * w_{k, t+1})` to build a robust global model without ever seeing the raw data.
*   **Security:** Before joining a training round, every node must provide a valid token issued by the `BlockchainAuth` module, simulating a secure ledger-based verification.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
