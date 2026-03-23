import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import hashlib
import time
import copy
import os
import time

from model import TransCRL
from datasets import CSIDataset
from federated_logic import fed_avg, LocalUpdate
from blockchain_auth import BlockchainAuth

def validate_model(model, dataloader, device):
    """Validate the model and return loss and accuracy"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    # 1. Configuration
    config = {
        "num_clients": 5,           # Number of IoT nodes 
        "global_rounds": 10,        # Total communication rounds 
        "local_epochs": 5,          # Epochs per client per round [cite: 72]
        "batch_size": 16,           # Training batch size
        "learning_rate": 0.0005,    # Initial learning rate
        "fedprox_mu": 0.01,         # Proximal term constant (0 for FedAvg) 
        "data_samples": 1200,       # Total samples for simulation [cite: 61]
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    device = config["device"]
    print("Using device:", device)
    
    # 2. Blockchain Authentication (B-RMA)
    b_auth = BlockchainAuth()
    
    for i in range(config["num_clients"]):
        client_id = f"client_{i}"
        pub_key = f"PUB_KEY_{i}"
        
        # Step 1: Registration (The 'Trust Gate' in BAN Logic)
        b_auth.register_node(client_id, pub_key)
        
        # Step 2: Initiate Challenge (Freshness Nonce)
        challenge = b_auth.initiate_challenge(client_id)
        
        # Step 3: Generate and Verify Response (Mutual Authentication)
        # In production, the client does this; here we simulate the signature
        mock_sig = hashlib.sha256(f"{challenge}{pub_key}".encode()).hexdigest()
        
        if b_auth.verify_response(client_id, mock_sig):
            print(f"Client {i} successfully joined the network.")
        else:
            print(f"Failed to authenticate client {i}. Exiting.")
            return

    # 3. Setup Dataset
    # Simulate a global dataset and split it for clients
    global_dataset = CSIDataset(data_dir="./datasets", is_synthetic=True, n_samples=900)
    
    # Create train/validation split (80/20)
    train_size = int(0.8 * len(global_dataset))
    val_size = len(global_dataset) - train_size
    train_dataset, val_dataset = random_split(global_dataset, [train_size, val_size])
    
    # Split training data among clients
    client_data_size = len(train_dataset) // config["num_clients"]
    client_datasets = random_split(train_dataset, [client_data_size] * config["num_clients"])
    
    client_loaders = [DataLoader(ds, batch_size=config["batch_size"], shuffle=True) for ds in client_datasets]
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 4. Initialize Global Model and Logging
    global_model = TransCRL().to(device)
    writer = SummaryWriter(log_dir="./logs")
    
    print("\nStarting Federated Training...")
    
    # 5. Federated Training Loop
    for round_idx in range(config["global_rounds"]):
        round_start_time = time.time()
        local_weights = []
        local_losses = []
        local_samples = []
        
        print(f"\n--- Global Round {round_idx + 1} ---")
        
        for i in range(config["num_clients"]):
            client_id = f"client_{i}"
            
            # Verify authorization via blockchain simulation before each round
            if not b_auth.is_authorized(client_id):
                print(f"Skipping unauthorized client: {client_id}")
                continue
                
            print(f"Local training on {client_id}...")
            
            # Local model starts with current global weights
            local_model = copy.deepcopy(global_model)
            trainer = LocalUpdate(dataloader=client_loaders[i], device=device, epochs=config["local_epochs"], lr=config["learning_rate"])
            
            new_weights, avg_loss, epoch_losses = trainer.train(local_model)
            
            local_weights.append(copy.deepcopy(local_model))
            local_losses.append(avg_loss)
            local_samples.append(len(client_datasets[i]))
            
            # Log per-client metrics
            writer.add_scalar(f'Loss/Client_{i}', avg_loss, round_idx)
            writer.add_scalar(f'Samples/Client_{i}', len(client_datasets[i]), round_idx)
            
            # Log per-epoch losses for this client
            for epoch_idx, epoch_loss in enumerate(epoch_losses):
                writer.add_scalar(f'Loss/Client_{i}/Epoch', epoch_loss, round_idx * config["local_epochs"] + epoch_idx)

        # 6. Weight Aggregation (FedAvg)
        global_model = fed_avg(global_model, local_weights, local_samples)
        
        # 7. Validation
        val_loss, val_accuracy = validate_model(global_model, val_loader, device)
        
        # 8. Logging results
        round_time = time.time() - round_start_time
        avg_round_loss = sum(local_losses) / len(local_losses) if local_losses else 0
        
        # TensorBoard logging
        writer.add_scalar('Loss/Global_Train', avg_round_loss, round_idx)
        writer.add_scalar('Loss/Global_Validation', val_loss, round_idx)
        writer.add_scalar('Accuracy/Global_Validation', val_accuracy, round_idx)
        writer.add_scalar('Time/Round_Training', round_time, round_idx)
        writer.add_scalar('Hyperparameters/Learning_Rate', config["learning_rate"], round_idx)
        
        # Log model histograms (every few rounds to avoid overhead)
        if round_idx % 2 == 0:
            for name, param in global_model.named_parameters():
                writer.add_histogram(f'Weights/{name}', param, round_idx)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, round_idx)
        
        print(f"Round {round_idx + 1} - Train Loss: {avg_round_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Time: {round_time:.2f}s")
        
        # Log client-specific losses for this round
        for i, loss in enumerate(local_losses):
            writer.add_scalar(f'Loss/Round_{round_idx}/Client_{i}', loss, round_idx)

    # 9. Finalize
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(global_model.state_dict(), "./models/global_model.pth")
    
    # Final validation
    final_val_loss, final_val_accuracy = validate_model(global_model, val_loader, device)
    writer.add_scalar('Loss/Final_Validation', final_val_loss, config["global_rounds"])
    writer.add_scalar('Accuracy/Final_Validation', final_val_accuracy, config["global_rounds"])
    
    writer.close()
    print("\nTraining completed. Global model saved to ./models/global_model.pth")
    print(f"Final Validation - Loss: {final_val_loss:.4f}, Accuracy: {final_val_accuracy:.2f}%")

if __name__ == "__main__":
    main()
