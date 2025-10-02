# src/metrics/calibration.py
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, model, dataloader, device):
        """
        Finds the optimal temperature by minimizing NLL on a validation set.
        """
        model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                all_logits.append(logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        # Create a temperature parameter to optimize
        temperature = nn.Parameter(torch.ones(1).to(device))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(all_logits / temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        self.temperature = temperature.item()
        print(f"Optimal temperature found: {self.temperature:.4f}")
        return self.temperature
    
def calculate_ece(posteriors, labels, n_bins=15):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    confidences, predictions = torch.max(posteriors, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = torch.zeros(1, device=posteriors.device)
    for i in range(n_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()
