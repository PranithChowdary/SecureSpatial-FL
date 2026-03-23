import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from preprocessing import low_pass_filter, cfr_modeling, extract_periodogram

class CSIDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_synthetic=False, n_samples=100):
        self.data_dir = data_dir
        self.transform = transform
        self.is_synthetic = is_synthetic
        
        if is_synthetic:
            self.data = [np.random.randn(2000, 30) + 1j * np.random.randn(2000, 30) for _ in range(n_samples)]
            # Labels: 0 (No presence), 1 (Presence)
            self.labels = np.random.randint(0, 2, n_samples)
        else:
            # Placeholder for loading actual data from files (Widar 3.0 or BGL)
            self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat') or f.endswith('.csv')]
            # Assume labels are in filenames or a separate mapping
            # self.labels = ...
            pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        csi_raw = self.data[idx]
        label = self.labels[idx]
        
        # Apply preprocessing
        dynamic_csi = cfr_modeling(csi_raw)
        _, _, spec = extract_periodogram(np.abs(dynamic_csi))
        
        # Convert spec to tensor (C, H, W)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
            
        return spec_tensor, torch.tensor(label, dtype=torch.long)

def download_datasets(target_dir="./datasets"):
    """
    Placeholder function to automate downloading datasets.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    print("Download logic for Widar3.0/BGL goes here.")

if __name__ == "__main__":
    download_datasets()
    dataset = CSIDataset(data_dir="./datasets", is_synthetic=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for inputs, labels in loader:
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break
