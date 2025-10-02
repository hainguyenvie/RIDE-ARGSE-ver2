# src/models/experts.py
import torch
import torch.nn as nn
from .backbones.resnet_cifar import CIFARResNet32

class Expert(nn.Module):
    """
    Expert model for CIFAR datasets using CIFARResNet-32 backbone.
    
    This implementation uses a CIFAR-optimized ResNet-32 architecture
    designed specifically for 32x32 images with proper initialization
    and optional regularization.
    """
    
    def __init__(self, num_classes=100, backbone_name='cifar_resnet32', 
                dropout_rate=0.0, init_weights=True):
        super(Expert, self).__init__()
        
        # Initialize backbone
        if backbone_name == 'cifar_resnet32' or backbone_name == 'resnet32':
            self.backbone = CIFARResNet32(
                dropout_rate=dropout_rate, 
                init_weights=init_weights
            )
            # Get feature dimension dynamically from backbone
            feature_dim = self.backbone.get_feature_dim()
            self.fc = nn.Linear(feature_dim, num_classes)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not recognized. "
                        f"Supported: 'cifar_resnet32', 'resnet32'")
        
        # Temperature scaling parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        
        # Initialize classifier if backbone init is enabled
        if init_weights:
            self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the final classifier layer."""
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """Forward pass through backbone and classifier."""
        features = self.backbone(x)
        logits = self.fc(features)
        return logits
    
    def get_calibrated_logits(self, x):
        """Get temperature-scaled logits for calibrated predictions."""
        return self.forward(x) / self.temperature
    
    def get_features(self, x):
        """Extract features from the backbone without classification."""
        return self.backbone(x)

    def set_temperature(self, temp):
        """Set temperature scaling parameter."""
        self.temperature.data = torch.tensor(temp, dtype=self.temperature.dtype)
    
    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """Print model summary."""
        total_params = self.get_num_parameters()
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        
        print("Expert Model Summary:")
        print(f"  Backbone: {backbone_params:,} parameters")
        print(f"  Classifier: {classifier_params:,} parameters") 
        print(f"  Total: {total_params:,} parameters")
        print(f"  Temperature: {self.temperature.item():.4f}")