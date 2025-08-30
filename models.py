import torch
import utils

from torchvision import models
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# EfficientNetB2
def create_effnetb2(num_classes: int,
                    device: str,
                    seed: int=42):
    
    utils.set_seed(seed)
    model_weights = models.EfficientNet_B2_Weights.DEFAULT
    model = models.efficientnet_b2(weights=model_weights).to(device)

    # Freezing Layers
    for param in model.parameters():
        param.requires_grad = False

    # Modifying the classifier layer according to number of classes required
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408,
                  out_features=num_classes)
    ).to(device)

    # Getting Transforms from model weights
    transforms = model_weights.transforms()

    return model, transforms

# ResNet50
def create_resnet50(num_classes: int,
                    device: str,
                    seed: int=42):
    
    utils.set_seed(seed)
    model_weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=model_weights).to(device)

    # Freezing all layers but last one
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

        else:
            param.requires_grad = False

    model.fc = nn.Linear(in_features=2048,
                        out_features=num_classes).to(device)

    # Getting Transforms from model weights
    transforms = model_weights.transforms()

    return model, transforms

# ResNET34
def create_resnet34(num_classes: int,
                    device: str,
                    seed: int=42):
    
    utils.set_seed(seed)
    model_weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=model_weights).to(device)

    # Freezing all layers but last one
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Sequential(   #type: ignore
        nn.Dropout(p=0.5),
        nn.Linear(in_features=512,
                out_features=num_classes)
    ).to(device)

    # Getting Transforms from model weights
    transforms = model_weights.transforms()

    return model, transforms
