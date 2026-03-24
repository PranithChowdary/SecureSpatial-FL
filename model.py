import torch
import torch.nn as nn

class TransCRL(nn.Module):
    def __init__(self, num_classes=2):
        super(TransCRL, self).__init__()
        
        # 1. CNN Module: Extracts spatial features from 2D periodograms
        # Input shape: (Batch, 1, Freq_bins, Time_steps) -> (B, 1, 33, 63)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # Added for training stability
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Downsample frequency, keep time resolution
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # Added to prevent overfitting
            nn.MaxPool2d(kernel_size=(2, 1)) # Result: (B, 32, 8, 63)
        )
        
        # 2. Transformer Module: Captures signal fluctuations over time
        # We treat the height (frequency features) * channels as the feature vector 
        # for each time step in the sequence.
        self.d_model = 32 * 8 # 256
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        
        # 3. LSTM Module: Models the temporal evolution of human gait
        self.lstm = nn.LSTM(
            input_size=self.d_model, 
            hidden_size=128, 
            num_layers=1, 
            batch_first=True
        )
        
        # 4. Final Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Added to prevent overfitting on small CSI datasets
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Freq, Time) -> (B, 1, 33, 63)
        batch_size = x.size(0)
        
        # Phase 1: CNN Spatial Feature Extraction
        x = self.cnn(x)  # Output: (B, 32, 8, 63)
        
        # Phase 2: Prepare for Transformer (Temporal Sequence)
        # We want the sequence length to be the 'Time' dimension (63)
        # Reshape to (B, Channels * Freq_features, Time)
        x = x.view(batch_size, self.d_model, -1) # (B, 256, 63)
        x = x.transpose(1, 2) # (B, 63, 256) -> (Batch, Seq_Len, Feature_Dim)
        
        # Phase 3: Transformer Self-Attention
        # Captures long-range dependencies across the time steps
        x = self.transformer_encoder(x) # (B, 63, 256)
        
        # Phase 4: LSTM Temporal Modeling
        # Extracts gait patterns from the attention-weighted sequence
        x, (h_n, c_n) = self.lstm(x) # x: (B, 63, 128)
        
        # Phase 5: Classification
        # Use the final hidden state which represents the entire sequence summary
        out = h_n[-1] # (B, 128)
        out = self.fc(out) # (B, num_classes)
        
        return out

if __name__ == "__main__":
    # Test with dummy tensor (B, C, Freq, Time)
    # Based on extract_periodogram output from preprocessing.py
    dummy_input = torch.randn(8, 1, 33, 63)
    model = TransCRL()
    output = model(dummy_input)
    print(f"Model input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")