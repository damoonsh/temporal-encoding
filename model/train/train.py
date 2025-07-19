import torch

class TrainModule:
    ''' Train Module for both JEPA encoder training and the forecaster '''

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
    
    def jepa_step(self, train=True):
        if train: 
            self.model.train()
        else:
            self.model.eval()

        running_train_loss = 0.0
        loader = self.train if train else self.val
        for inputs in loader:
            inputs = inputs.to(self.device), inputs.to(self.device)
                
            if train: optimizer.zero_grad()
            output = model(inputs)
            loss = torch.nn.functional.mse_loss(output, y)
                
            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(loader.dataset)

        if train:
            self.train_losses['jepa'].append(epoch_loss)
        else:
            self.val_losses['jepa'].append(epoch_loss)

        if not train: self.scheduler.step(epoch_loss)

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

    def jepa_train(self, log=True):
        for epoch in range(self.num_epochs):
            self.jepa_step()
            self.jepa_step(train=False)
            self.log(epoch, self.train_losses['forecast'][-1], self.val_losses['forecast'][-1])
            if not self.es_save: break
