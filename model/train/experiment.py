import itertools

def generate_param_combinations(common_params, possible_params):
    # Generate all possible combinations of values from possible_params
    param_keys = list(possible_params.keys())
    param_values = [possible_params[key] for key in param_keys]
    all_combinations = list(itertools.product(*param_values))
    
    # Create list to store all parameter dictionaries
    result = []
    
    # For each combination, create a new dictionary
    for combo in all_combinations:
        # Start with common_params
        new_params = common_params.copy()
        
        # Update with current combination of possible params
        for key, value in zip(param_keys, combo):
            new_params[key] = value
            
        # Generate model name based on variable parameters
        new_params['model_name'] = f"s{new_params['seq_len']}"
            
        result.append(new_params)
    
    return result


# Define parameters
common_params = {
    'patch_len': 16,
    'num_patch': 5,
    'seq_len': 32,
    'seq_stride': 3,
    
    'val_indices': ['NYA'],
    'test_indices': ['GDAXI'],
    'num_epochs': 150,
    'early_stopping_patience': 30,
    'mask_ratio': 0.7,
    "mamba_encoder_embed_dim": 8,'forecast':False ,'rev': False, 'bw': False
}

def evaluate_encoding_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    running_test_loss = 0.0
    total_samples = 0

    model.to(device)
    with torch.no_grad(): 
        for inputs in test_loader:
            inputs = inputs.to(device)
            loss = model(inputs)
            running_test_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    if total_samples == 0:
        print("Warning: Test loader is empty.")
        return 0.0 # Avoid division by zero

    average_test_loss = running_test_loss / total_samples
    print(f"Average Test Loss: {average_test_loss:.4f}")
    return average_test_loss

def evaluate_simplef_model(model, test_loader, device):
    model.eval()
    running_test_loss = 0.0
    total_samples = 0

    model.to(device)
    with torch.no_grad():
         for X, (mean, std), y in test_loader:
            X, y = X.to(device), y.to(device)
            mean, std = mean.to(device), std.to(device)
            output = model(X)
            output = output * std.unsqueeze(1) + mean.unsqueeze(1)
            loss = torch.nn.functional.mse_loss(output, y)
            running_test_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

    if total_samples == 0:
        print("Warning: Test loader is empty.")
        return 0.0

    average_test_loss = running_test_loss / total_samples
    print(f"Average Test Loss: {average_test_loss:.4f}")
    return average_test_loss

def train_param(t):
    seq_len, seq_stride = t['seq_len'], t['seq_stride']
    pl, np = t['patch_len'], t['num_patch']
    BATCH_SIZE = t.get('batch_size', 32)
    
    return_y = t['forecast']

    # Dataset construction with return_y
    train_dataset = IndexData(result, 'Open', 'all', seq_len, seq_stride, pl, np, return_y=return_y, batch_wise=t['bw'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dataset = IndexData(index, 'Open', t['val_indices'], seq_len, seq_stride, pl, np, return_y=return_y, batch_wise=t['bw'])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = IndexData(index, 'Open', t['test_indices'], seq_len, seq_stride, pl, np, return_y=return_y, batch_wise=t['bw'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if return_y:
        # Use encoder from t and SimpleF
        scale = t.get('scale', True)
        model = t['infer_model'](encoder=t['encoder'])
        model, train_loss, val_loss = train_forecast_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=t['num_epochs'],
            early_stopping_patience=t['early_stopping_patience'],
            model_name=t['model_name'],
            device=device, scale=scale
        )
        test_loss = evaluate_simplef_model(model, test_loader, device)
    else:
        jepa_model = JEPA(
            encoder=MambaEncoder(), data_encoder=TempEncoder(),
            predictor=BaseJepaPredictor(), rev=t['rev']
        )
        model, train_loss, val_loss = train_jepa_model(
            model=jepa_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=t['num_epochs'],
            early_stopping_patience=t['early_stopping_patience'],
            model_name=t['model_name'],
            device=device
        )
        test_loss = evaluate_encoding_model(model, test_loader, device)

    return model, train_loss, val_loss, test_loss

def run_combinations(combs):
    model_metrics = {}
    for param in combs:
        model, train_loss, val_loss, test_loss = train_param(param)
        model_metrics[param['model_name']] = {'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
    
    import pickle
    with open('/kaggle/working/train_metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)

    return model, model_metrics

