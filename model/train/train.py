def train_jepa_model(model, train_loader, val_loader, num_epochs, early_stopping_patience, model_name, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=16, verbose=False)
    best_val_loss = float('inf')
    best_epoch = None
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
            loss = model(inputs)
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
                loss = model(inputs)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)  # Append validation loss

        print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(epoch_val_loss)

        # --- Early Stopping & Model Saving ---
        if epoch_val_loss < best_val_loss:
            # print(f"Validation loss decreased ({best_val_loss:.4f} --> {epoch_val_loss:.4f}). Saving model ...")
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            # Use deepcopy to ensure the state dict is not modified later
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_name)
            patience_counter = 0
        else:
            patience_counter += 1
            # print(f"Validation loss did not improve. Early Stopping Counter: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                # print("Early stopping triggered.")
                break

    print("Training finished.")
    # Optionally load the best model state before returning
    if best_model_state:
        print(f"Loading best model at epoch {best_epoch}")
        model.load_state_dict(torch.load(model_name, weights_only=True))
    else:
        print("No best model state was saved (validation loss never improved).")

    return model, train_losses, val_losses 
    

def train_forecast_model(model, train_loader, val_loader, num_epochs, early_stopping_patience, model_name, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=10, verbose=False)
    best_val_loss = float('inf')
    best_epoch = None
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for X, (mean, std), y in train_loader:
            X, y = X.to(device), y.to(device)
            mean, std = mean.to(device), std.to(device)
            optimizer.zero_grad()
            output = model(X)
            # print(y[0], output[0])
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * X.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X, (mean, std), y in train_loader:
                X, y = X.to(device), y.to(device)
                mean, std = mean.to(device), std.to(device)
                output = model(X)
                if scale:  output = output * std.unsqueeze(1) + mean.unsqueeze(1)
                loss = torch.nn.functional.mse_loss(output, y)
                running_val_loss += loss.item() * X.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_train_loss > 1e6:
            print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss/1e6:.4f}, Val loss: {epoch_val_loss/1e6:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}")

        scheduler.step(epoch_val_loss)

        # --- Early Stopping & Model Saving ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, model_name)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    print("Training finished.")
    if best_model_state:
        print(f'Best at {best_epoch}')
        model.load_state_dict(torch.load(model_name, map_location=device))
    else:
        print("No best model state was saved (validation loss never improved).")

    return model, train_losses, val_losses

class TrainJepa:

    def __init__(self, model, train_loader, num_epochs, early_stopping_patience, model_name, device):
        self.model = model
        self.train = train_loader
        self.val = val_loader
        self.num_epochs = num_epochs
        self.esp = early_stopping_patience

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.99, patience=early_stopping_patience, verbose=False)

        self.best_val_loss = float('inf')
        self.best_epoch = None
        self.patience_counter = 0
        self.best_model_state = None

        self.train_losses = {'forecast': [], 'jepa': []}
        self.val_losses = {'forecast': [], 'jepa': []}

    def log(self, epoch, epoch_train_loss, epoch_val_loss):
        if epoch_train_loss > 1e6:
            print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss/1e6:.4f}, Val loss: {epoch_val_loss/1e6:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Train loss: {epoch_train_loss:.4f}, Val loss: {epoch_val_loss:.4f}")
    
    def forecast_step(self, train=True):
        ''' Forecast step: Goes through epoch, tracks metrics, and can handle both the train and val'''
        if train: 
            self.model.train()
        else:
            self.model.eval()

        running_train_loss = 0.0
        loader = self.train if train else self.val
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
                
            if train: optimizer.zero_grad()
            output = model(X)
            loss = torch.nn.functional.mse_loss(output, y)
                
            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(loader.dataset)

        if train:
            self.train_losses['forecast'].append(epoch_loss)
        else:
            self.val_losses['forecast'].append(epoch_loss)

        if not train: self.scheduler.step(epoch_loss)
    
    def es_save(self, epoch_val_loss, model_name):
        if epoch_val_loss < self.best_val_loss:
            self.best_val_loss = epoch_val_loss
            self.best_epoch = epoch
            self.best_model_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_model_state, model_name)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.esp: return True
        
        return False

    def forecast_train(self, log=True):
        for epoch in range(self.num_epochs):
            self.forecast_step()
            self.forecast_step(train=False)
            self.log(epoch, self.train_losses['forecast'][-1], self.val_losses['forecast'][-1])
            if not self.es_save: break
