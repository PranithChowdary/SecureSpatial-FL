import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter  # Disabled due to environment issues
import copy
import os

from model import TransCRL
from datasets import CSIDataset
from federated_logic import fed_avg, LocalUpdate
from blockchain_auth import BlockchainAuth

def main():
    # 1. Configuration
    num_clients = 3
    global_rounds = 5
    local_epochs = 3
    batch_size = 8
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Blockchain Authentication (B-RMA)
    b_auth = BlockchainAuth()
    client_auth_tokens = {}
    
    for i in range(num_clients):
        client_id = f"client_{i}"
        token = b_auth.register_node(client_id, f"PUB_KEY_{i}")
        if b_auth.authenticate_node(client_id, token):
            client_auth_tokens[client_id] = token
        else:
            print(f"Failed to authenticate client {i}. Exiting.")
            return

    # 3. Setup Dataset
    # Simulate a global dataset and split it for clients
    global_dataset = CSIDataset(data_dir="./datasets", is_synthetic=True, n_samples=300)
    client_data_size = len(global_dataset) // num_clients
    client_datasets = random_split(global_dataset, [client_data_size] * num_clients)
    
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    
    # 4. Initialize Global Model and Logging
    global_model = TransCRL().to(device)
    # writer = SummaryWriter(log_dir="./logs")
    writer = None
    
    print("Starting Federated Training...")
    
    # 5. Federated Training Loop
    for round_idx in range(global_rounds):
        local_weights = []
        local_losses = []
        local_samples = []
        
        print(f"\n--- Global Round {round_idx + 1} ---")
        
        for i in range(num_clients):
            client_id = f"client_{i}"
            
            # Verify authorization via blockchain simulation before each round
            if not b_auth.is_authorized(client_id):
                print(f"Skipping unauthorized client: {client_id}")
                continue
                
            print(f"Local training on {client_id}...")
            
            # Local model starts with current global weights
            local_model = copy.deepcopy(global_model)
            trainer = LocalUpdate(dataloader=client_loaders[i], device=device, epochs=local_epochs, lr=learning_rate)
            
            new_weights, avg_loss = trainer.train(local_model)
            
            local_weights.append(copy.deepcopy(local_model))
            local_losses.append(avg_loss)
            local_samples.append(len(client_datasets[i]))
            
        # 6. Weight Aggregation (FedAvg)
        global_model = fed_avg(global_model, local_weights, local_samples)
        
        # Logging results
        avg_round_loss = sum(local_losses) / len(local_losses)
        if writer:
            writer.add_scalar('Loss/Global_Round', avg_round_loss, round_idx)
        print(f"Round {round_idx + 1} average loss: {avg_round_loss:.4f}")

    # 7. Finalize
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(global_model.state_dict(), "./models/global_model.pth")
    print("\nTraining completed. Global model saved to ./models/global_model.pth")
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
