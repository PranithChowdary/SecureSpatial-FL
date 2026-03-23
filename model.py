import torch
import torch.nn as nn

class TransCRL(nn.Module):
    def __init__(self, num_classes=2):
        super(TransCRL, self).__init__()
        
        # 1. CNN Module: Extract spatial features from 2D periodograms
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 2. Transformer Module: Capture signal fluctuations
        # Assume input to transformer is (Batch, Sequence_Len, Hidden_Dim)
        # We'll treat the spatial features from CNN as a sequence
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        
        # 3. LSTM Module: Model temporal evolution
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        
        # 4. Final Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 1, H, W) where H x W is the periodogram size
        batch_size = x.size(0)
        
        # CNN
        x = self.cnn(x)  # (B, 32, H', W')
        
        # Flatten spatial dimensions to form a sequence for Transformer/LSTM
        # Let's flatten H' and W' but keep the channel dimension as hidden_dim
        # Alternatively, use one spatial dimension as 'time' if applicable
        # Here we'll pool and flatten for simplicity in this prototype
        x = x.view(batch_size, 32, -1)  # (B, 32, Seq_Len)
        x = x.permute(2, 0, 1)  # (Seq_Len, B, 32) for Transformer
        
        # Transformer
        x = self.transformer_encoder(x)  # (Seq_Len, B, 32)
        
        # LSTM
        x = x.permute(1, 0, 2)  # (B, Seq_Len, 32)
        x, (h_n, c_n) = self.lstm(x)  # x: (B, Seq_Len, 64)
        
        # Use the last hidden state of LSTM for classification
        out = h_n[-1]  # (B, 64)
        out = self.fc(out)  # (B, num_classes)
        
        return out

if __name__ == "__main__":
    # Test with dummy tensor (B, C, H, W)
    # Based on extract_periodogram spec shape (33, 63)
    dummy_input = torch.randn(8, 1, 33, 63)
    model = TransCRL()
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
