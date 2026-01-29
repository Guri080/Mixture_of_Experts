import torchvision
import torch.nn as nn
from torvision.models import models


def Swin_base(in_channels, num_classes, pretrained=True):
    """
    Creates a Swin Base model that adapts automatically to input channel size and output classes.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load ImageNet-pretrained weights
    
    Returns:
        model (torch.nn.Module): Adapted Swin Transformer model
    """
    
    # Load pretrained or random-initialized model
    if pretrained:
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_b(weights=None)
    
    # --- Adapt first conv if input channels != 3 ---
    first_conv = model.features[0][0]
    if in_channels != first_conv.in_channels:
        # Create new conv layer with same hyperparams but different input channels
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        
        # If pretrained weights are available and input has 1 channel,
        # copy and average RGB weights to initialize
        if pretrained and in_channels == 1:
            new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        
        model.features[0][0] = new_conv
    
    # --- Replace classification head ---
    in_ftrs = model.head.in_features
    model.head = nn.Linear(in_ftrs, num_classes)
    
    return model