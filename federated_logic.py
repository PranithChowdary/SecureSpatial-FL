import torch
import copy

def fed_avg(global_model, local_models, local_n_samples):
    """
    Standard Federated Averaging (FedAvg) algorithm[cite: 69, 70, 71].
    w_{t+1} = sum (n_k/n * w_{k, t+1})
    """
    total_samples = sum(local_n_samples)
    global_weights = global_model.state_dict()
    aggregated_weights = copy.deepcopy(global_weights)
    
    for key in aggregated_weights.keys():
        # Initialize only floating point tensors to zero for aggregation
        if aggregated_weights[key].is_floating_point():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
        
    for i, model in enumerate(local_models):
        local_weights = model.state_dict()
        weight_factor = local_n_samples[i] / total_samples
        for key in local_weights.keys():
            if aggregated_weights[key].is_floating_point():
                aggregated_weights[key] += local_weights[key] * weight_factor
            else:
                # For non-floating point (Long) tensors like num_batches_tracked,
                # just take the value from the first local model or keep global
                if i == 0:
                    aggregated_weights[key] = local_weights[key]
            
    global_model.load_state_dict(aggregated_weights)
    return global_model

class LocalUpdate:
    def __init__(self, dataloader, device, epochs=5, lr=0.01, mu=0.01):
        """
        Args:
            mu (float): Proximal term constant for FedProx. 
                        Set mu=0 to revert to standard FedAvg.
        """
        self.dataloader = dataloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.mu = mu 
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model):
        # Keep a copy of the global model weights for the FedProx proximal term
        global_weight_collector = copy.deepcopy(model.state_dict())
        
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
                
                # Standard Cross Entropy Loss
                base_loss = self.criterion(outputs, labels)
                
                # FedProx Proximal Term: (mu/2) * ||w - w_t||^2
                # This penalizes local updates that stray too far from the global model
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += (param - global_weight_collector[name].to(self.device)).norm(2)**2
                
                loss = base_loss + (self.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
        return model.state_dict(), sum(epoch_losses) / len(epoch_losses), epoch_losses

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
