import torch
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device,):   #type: ignore

    '''Training Loop for training te neural network

    Args:
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fn,
        optimizer: torch.optim.Optimizer,
        device: torch.device   
    '''

    model.train()
    train_loss = 0
    train_accuracy = 0

    #Iterating through a Batch
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward Pass
        y_pred = model(X)
        # Loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # Accuracy
        accuracy = accuracy_fn(y, y_pred.argmax(dim=1))
        # train_accuracy += accuracy
        labels = torch.argmax(y_pred, dim=1)
        # correct_mask = (labels == y)
        # accuracy = torch.sum(correct_mask).float() / len(y)
        train_accuracy += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)

    return train_loss, train_accuracy


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device): # type: ignore

    '''Testing Loop for neural network

    Args:
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fn,
        device: torch.device   
    
    Returns:
        test_loss, test_accuracy
    '''

    model.eval()
    test_loss = 0
    test_accuracy = 0
    with(torch.inference_mode()):
        for X_test, y_test in dataloader:
            X_test, y_test = X_test.to(device), y_test.to(device)

            test_pred = model(X_test)
            loss = loss_fn(test_pred, y_test)
            test_loss += loss
            accuracy = accuracy_fn(y_test, test_pred.argmax(dim=1))
            # test_accuracy += accuracy
            labels = torch.argmax(test_pred, dim=1)
            # correct_mask = (labels == y_test)
            # accuracy = torch.sum(correct_mask).float() / len(y_test)
            test_accuracy += accuracy


        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

        return test_loss, test_accuracy


def train_model(model,
                train_dataloader,
                test_dataloader,
                loss_fn,
                accuracy_fn,
                optimizer,
                epochs: int = 5,
                device=device,
                scheduler_fn = None,):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           loss_fn=loss_fn,
                                           accuracy_fn=accuracy_fn,
                                           device=device) # type: ignore

        test_loss, test_acc = test_step(model=model,
                                           dataloader=test_dataloader,
                                           loss_fn=loss_fn,
                                           accuracy_fn=accuracy_fn,
                                           device=device) # type: ignore

        if scheduler_fn:
            scheduler_fn.step(test_loss)

        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.3f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.3f}%")

        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

# Making predictions on test dataset
def make_predictions(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device):
    """
    Runs inference on a given dataloader and returns predicted and true labels.

    Args:
        model (torch.nn.Module): Trained PyTorch model for making predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data to predict on.
        device (torch.device): Device to perform computation on ('cpu' or 'cuda').

    Returns:
        Tuple(torch.Tensor, torch.Tensor): 
            - eval_preds_tensor: Concatenated tensor of predicted class indices.
            - target_labels_tensor: Concatenated tensor of true class labels.
    """
    eval_preds = []
    target_labels = []
    model.eval()
    with(torch.inference_mode()):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            eval_logits = model(X)
            eval_pred = torch.softmax(eval_logits, dim=0).argmax(dim=1)
            eval_preds.append(eval_pred.cpu())
            target_labels.append(y.cpu())

        # Creating a single tensor
        eval_preds_tensor = torch.cat(eval_preds)
        target_labels_tensor = torch.cat(target_labels)
        
        return eval_preds_tensor, target_labels_tensor