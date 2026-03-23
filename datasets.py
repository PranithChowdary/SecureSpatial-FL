import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from preprocessing import cfr_modeling, extract_periodogram

class CSIDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_synthetic=False, n_samples=1000, force_preprocess=False):
        """
        Args:
            data_dir (string): Directory with all the CSI files.
            is_synthetic (bool): Generate dummy data for prototyping.
            force_preprocess (bool): If True, re-runs FFT on raw data even if processed files exist.
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.transform = transform
        self.is_synthetic = is_synthetic
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if is_synthetic:
            self._generate_synthetic(n_samples)
        else:
            self._prepare_real_data(force_preprocess)

    def _generate_synthetic(self, n_samples):
        """Generates synthetic processed tensors for rapid prototyping."""
        print("Generating synthetic dataset...")
        self.samples = []
        self.labels = np.random.randint(0, 2, n_samples)
        for i in range(n_samples):
            # Shape matches (1, Frequency_Bins, Time_Steps) from preprocessing.py
            dummy_spec = torch.randn(1, 33, 63) 
            self.samples.append(dummy_spec)

    def _prepare_real_data(self, force_preprocess):
        """
        Handles Widar3.0/BGL data. 
        Converts raw .dat/.csv files into processed PyTorch tensors (.pt).
        """
        # Find all raw CSI files 
        raw_files = glob.glob(os.path.join(self.data_dir, "*.dat")) + \
                    glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        self.processed_files = []
        self.labels = []

        print(f"Checking for processed data in {self.processed_dir}...")
        
        for f_path in tqdm(raw_files, desc="Preprocessing CSI Data"):
            file_name = os.path.basename(f_path)
            save_path = os.path.join(self.processed_dir, file_name + ".pt")
            
            # Label extraction logic (Modify based on Widar3.0 naming convention) 
            # Example: 'user1_room1_presence_0.dat' -> label 1
            label = 1 if "presence" in file_name.lower() else 0
            
            if not os.path.exists(save_path) or force_preprocess:
                # 1. Load raw CSI [cite: 60]
                # Note: Replace with actual .dat reader logic for Widar3.0
                raw_data = np.random.randn(2000, 30) + 1j * np.random.randn(2000, 30) 
                
                # 2. CFR Modeling & Feature Extraction [cite: 59, 63]
                dynamic_csi = cfr_modeling(raw_data)
                _, _, spec = extract_periodogram(np.abs(dynamic_csi))
                
                # 3. Save as Tensor to avoid re-calculation [cite: 63]
                spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
                torch.save((spec_tensor, label), save_path)
            
            self.processed_files.append(save_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_synthetic:
            return self.samples[idx], torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Load pre-processed tensor from disk for speed
        spec_tensor, label = torch.load(self.processed_files[idx])
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
            
        return spec_tensor, torch.tensor(label, dtype=torch.long)

def setup_data_folders(base_dir="./datasets"):
    """Creates the necessary directory structure for SecureSpatial-FL[cite: 57]."""
    folders = [base_dir, os.path.join(base_dir, "processed"), "./logs", "./models"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created: {folder}")

if __name__ == "__main__":
    setup_data_folders()
    # Initialize dataset
    # For real data, place .dat files in ./datasets and set is_synthetic=False
    dataset = CSIDataset(data_dir="./datasets", is_synthetic=True, n_samples=500)
    
    print(f"Dataset ready with {len(dataset)} samples.")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for inputs, labels in loader:
        print(f"Input batch shape: {inputs.shape}") # Expect (Batch, 1, Freq, Time)
        break