import torch
import torch.nn as nn
import torch.nn.functional as F

import copy 

class DyTanh(nn.Module):
    def __init__(self, shape, mean=0.1, std=0.001):
        super(DyTanh, self).__init__()
        self.scale = nn.Parameter(torch.randn(shape) * std + mean)
        self.shift = nn.Parameter(torch.randn(shape) * std + mean)
        self.alpha = nn.Parameter(torch.randn(shape) * std + mean)

    def forward(self, x):
        return self.scale * torch.tanh(self.alpha * x) + self.shift
    
class DyTanhInstance(nn.Module):
    def __init__(self, embed_dim, mean=0.1, std=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        return self.scale * torch.tanh(self.alpha * x) + self.shift

class DyTanhBatch(nn.Module):
    def __init__(self, embed_dim, mean=0.1, std=0.001):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, embed_dim) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        return self.scale * torch.tanh(self.alpha * x) + self.shift

class DyTanhGroup(nn.Module):
    def __init__(self, embed_dim, num_groups=4, mean=0.1, std=0.001):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = embed_dim // num_groups
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        self.scale = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)
        self.shift = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)
        self.alpha = nn.Parameter(torch.randn(1, 1, num_groups, 1) * std + mean)

    def forward(self, x):
        # x: (bs, seq_len, embed_dim)
        # Reshape to (bs, seq_len, num_groups, group_size)
        x_grouped = x.view(x.size(0), x.size(1), self.num_groups, -1)
        # Apply group-wise scaling
        out = self.scale * torch.tanh(self.alpha * x_grouped) + self.shift
        # Reshape back
        return out.view_as(x)

    
class BaseJepaPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        dim1, dim2 = input_size, input_size // 2
        self.net = nn.Sequential( 
            nn.Linear(dim1, dim2),
            DyTanh(dim2),
            nn.Linear(dim2, dim1)
        )
        nn.init.normal_(self.net[0].weight, mean=0.0, std=0.001)
        nn.init.normal_(self.net[2].weight, mean=0.0, std=0.001)

    def forward(self, x):
        return self.net(x)
    

def train_jepa_model(model, train_loader, val_loader, num_epochs, early_stopping_patience, model_name, device):
    """
    Trains a PyTorch model with early stopping and tracks losses.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        criterion (torch.nn.Module): The loss function.
        num_epochs (int): Maximum number of epochs to train for.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
        model_name (str): Name used for saving the best model checkpoint (e.g., 'best_model.pth').
        device (torch.device): The device to train on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The trained model (with best weights loaded if improvement occurred).
            - train_losses (list): List of training losses per epoch.
            - val_losses (list): List of validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=10, verbose=False)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []  # Initialize list to store training losses
    val_losses = []    # Initialize list to store validation losses

    model.to(device)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            loss, _, _, _ = model(inputs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)  # Append training loss

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device)
                loss, _, _, _ = model(inputs)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)  # Append validation loss

        print(f"Epoch {epoch+1}/{num_epochs}.. "
              f"Train loss: {epoch_train_loss:.4f}.. "
              f"Val loss: {epoch_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # --- Early Stopping & Model Saving ---
        if epoch_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model ...")
            best_val_loss = epoch_val_loss
            # Use deepcopy to ensure the state dict is not modified later
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_name)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Early Stopping Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    # Optionally load the best model state before returning
    if best_model_state:
        print(f"Loading best model state saved to {model_name}")
        model.load_state_dict(torch.load(model_name))
    else:
        print("No best model state was saved (validation loss never improved).")

    return model, train_losses, val_losses  # Return model and loss lists