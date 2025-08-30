import os
import random
import torch
import torchvision
import zipfile
import requests
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from pathlib import Path
from typing import Dict, Tuple, List
from PIL import Image
from torchvision import transforms
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model: torch.nn.Module,
               target_dir: Path,
               model_name: str):
    """Saves a PyTorch model to a specified directory.

       Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        target_dir (str): The directory where the model will be saved.
        model_name (str): The name of the model file.

        Returns: None
    """

    # Create a target directory if it doesn't already exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Model name must end with .pth or .pt"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict() to a file
    print(f"[INFO] Saving PyTorch Model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


def plot_loss_curves(results: Dict[str, list[float]]):

    """
    Plots loss and accuracy curves for training and testing.

    Args:
        results (Dict[str, list[float]]): A dictionary containing training and testing
                                            loss and accuracy values.  Keys should be
                                            strings like 'train_loss', 'test_loss',
                                            'train_acc', and 'test_acc'.

    Returns:
        None: Plots the Loss and accuracy graphs
    """

    loss = results['train_loss']
    test_loss = results['test_loss']

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Getting number of epochs
    epochs = range(len(results['train_loss']))

    # Plotting
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.legend()


# Function to make predictions using the trained model on target image
def pred_and_plot_image(model: torch.nn.Module,
                        img_path: Path,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None, # type: ignore
                        device: torch.device = device):
    
    # Load the target image
    img = Image.open(img_path)

    # Create transformations if not provided
    if transform is not None:
        img_transforms = transform
    else:
        img_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    model.to(device)

    # Making predictions on the image
    transformed_img = img_transforms(img).unsqueeze(0) # type: ignore
    model.eval()
    with(torch.inference_mode()):
        pred = model(transformed_img.to(device))
    
    # Converting logits into class probabilities
    probs = torch.softmax(pred, dim=1)
    
    #Converting the predicted probabilities to a class labels
    predicted_class = probs.argmax(dim=1)

    # Plotting the image with labels and probabilities
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[predicted_class]} | Porb: {probs.max():.3f}")
    plt.axis(False)

def set_seed(seed: int=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None): # type: ignore
    """Creates a TensorBoard SummaryWriter instance at the specified directory.

    Args:
        experminent_name (str): The name of the experiment.
        model_name (str): The name of the model.
        extra (str, optional): Additional information to include in the summary writer. Defaults to None.

    Returns:
        SummaryWriter: A TensorBoard SummaryWriter instance.
    """

    # Timestamp for current date
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Creating logdirectory path
        log_dir = os.path.join("logs", experiment_name, model_name, timestamp, extra)
    else:
        log_dir = os.path.join("logs", experiment_name, model_name, timestamp)

    print(f"[INFO] Created SummaryWriter at {log_dir}")
    return SummaryWriter(log_dir=log_dir) 

# Plotting Confusion Matrix
def plot_confmat(preds: torch.Tensor,
                 target_labels: torch.Tensor,
                 class_names: List[str]):
    
    """Plots a confusion matrix given predictions and target labels.

    Args:
        preds (torch.Tensor): Predicted labels.
        target_labels (torch.Tensor): True labels.
        class_names (List[str]): A list of class names to use for the confusion matrix.

    Returns:
        None. Displays the confusion matrix plot.
    """

    confmat = ConfusionMatrix(task='multiclass',
                            num_classes=len(class_names))
    confmat_tensor = confmat(preds=preds,
                            target=target_labels)

    # Plotting 
    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(),
                                    class_names=class_names,
                                    figsize=(10, 7))