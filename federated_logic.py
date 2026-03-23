import torch
import copy

def fed_avg(global_model, local_models, local_n_samples):
    """
    Federated Averaging (FedAvg) algorithm.
    w_{t+1} = sum (n_k/n * w_{k, t+1})
    """
    total_samples = sum(local_n_samples)
    global_weights = global_model.state_dict()
    
    # Initialize aggregated weights to zero
    aggregated_weights = copy.deepcopy(global_weights)
    for key in aggregated_weights.keys():
        aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
    for i, model in enumerate(local_models):
        local_weights = model.state_dict()
        weight_factor = local_n_samples[i] / total_samples
        for key in local_weights.keys():
            aggregated_weights[key] += local_weights[key] * weight_factor
            
    global_model.load_state_dict(aggregated_weights)
    return global_model

class LocalUpdate:
    def __init__(self, dataloader, device, epochs=5, lr=0.01):
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model):
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        epoch_losses = []
        for epoch in range(self.epochs):
            batch_losses = []
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
        return model.state_dict(), sum(epoch_losses) / len(epoch_losses)

if __name__ == "__main__":
    from model import TransCRL
    from torch.utils.data import DataLoader, TensorDataset
    
    # Simple test for aggregation
    global_m = TransCRL()
    local_m1 = TransCRL()
    local_m2 = TransCRL()
    
    # Mocking different samples
    fed_avg(global_m, [local_m1, local_m2], [10, 20])
    print("Aggregation complete.")
