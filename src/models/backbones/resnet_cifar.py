# src/models/backbones/resnet_cifar.py
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNet-32."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv2d(
            in_planes, planes, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(
            planes, planes, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Apply dropout if specified
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Second conv block
        out = self.bn2(self.conv2(out))
        
        # Residual connection
        out += self.shortcut(x)
        
        # Final activation
        out = F.relu(out)
        return out

class CIFARResNet(nn.Module):
    """
    ResNet architecture optimized for CIFAR datasets (32x32 images).
    
    Key differences from ImageNet ResNet:
    - Smaller initial conv (3x3 instead of 7x7)
    - No max pooling after initial conv
    - Starts with 16 filters instead of 64
    - Uses adaptive average pooling for flexibility
    """
    
    def __init__(self, block, num_blocks, dropout_rate=0.0, init_weights=True):
        super(CIFARResNet, self).__init__()
        self.in_planes = 16
        self.dropout_rate = dropout_rate
        
        # Initial convolution - smaller for CIFAR
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate final feature dimension
        self.feature_dim = 64 * block.expansion
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()
        
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a residual layer with specified number of blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride_val in strides:
            layers.append(block(
                self.in_planes, planes, stride_val, self.dropout_rate
            ))
            self.in_planes = planes * block.expansion
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        # Initial conv and batch norm
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Global average pooling
        out = self.avgpool(out)
        
        # Flatten features
        out = out.view(out.size(0), -1)
        
        return out
    
    def get_feature_dim(self):
        """Return the dimension of the feature vector."""
        return self.feature_dim


def CIFARResNet32(dropout_rate=0.0, init_weights=True):
    """
    Construct a ResNet-32 for CIFAR datasets.
    
    Args:
        dropout_rate (float): Dropout rate for regularization (default: 0.0)
        init_weights (bool): Whether to initialize weights (default: True)
    
    Returns:
        CIFARResNet: ResNet-32 model optimized for CIFAR
    
    Architecture:
    - Total layers: 32 (1 + 3×(5×2) + 1 = 32)
    - 3 residual groups with 5 basic blocks each
    - Feature channels: 16 → 32 → 64
    - Output feature dimension: 64
    """
    return CIFARResNet(BasicBlock, [5, 5, 5], dropout_rate, init_weights)


# Backward compatibility alias
def ResNet32(dropout_rate=0.0, init_weights=True):
    """Backward compatibility alias for CIFARResNet32."""
    return CIFARResNet32(dropout_rate, init_weights)