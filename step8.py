# data_generation.py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def sample_waves(n_samples, frequency=40, n_points=10):
    """Generate sine and cosine wave samples.
    
    Args:
        n_samples (int): Number of wave samples to generate
        frequency (float): Wave frequency
        n_points (int): Number of points per wave
        
    Returns:
        tuple: (timestamps, sine_waves, cosine_waves)
    """
    step = frequency * n_points
    period = 1 / frequency
    start = np.random.uniform(0, period, size=n_samples)
    start = np.expand_dims(start, 1)
    t = np.arange(n_points) / step
    ts = start + t
    sines = np.sin(2*np.pi * frequency * ts)
    cosines = np.cos(2*np.pi * frequency * ts)
    return ts, sines, cosines

def prepare_data(n_samples=1000, batch_size=16, test_size=0.2, val_size=0.1):
    """Prepare data loaders for training, validation and testing.
    
    Args:
        n_samples (int): Number of samples to generate
        batch_size (int): Batch size for DataLoader
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of training data for validation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Generate data
    ts, X, y = sample_waves(n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_tensors(X, y):
        X_tensor = torch.from_numpy(X).float().unsqueeze(-1).to(device)
        y_tensor = torch.from_numpy(y).float().to(device)
        return X_tensor, y_tensor
    
    X_train_tensor, y_train_tensor = prepare_tensors(X_train, y_train)
    X_val_tensor, y_val_tensor = prepare_tensors(X_val, y_val)
    X_test_tensor, y_test_tensor = prepare_tensors(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# models.py
import torch.nn as nn

class WavePredictor(nn.Module):
    """Neural network for wave prediction using different RNN cell types."""
    
    def __init__(self, input_size, hidden_size=50, num_layers=2, 
                 output_size=1, bidirectional=False, cell_type="rnn"):
        """
        Args:
            input_size (int): Size of input features
            hidden_size (int): Number of hidden units
            num_layers (int): Number of RNN layers
            output_size (int): Size of output
            bidirectional (bool): Whether to use bidirectional RNN
            cell_type (str): Type of RNN cell ("rnn", "lstm", or "gru")
        """
        super(WavePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()
        
        # RNN layer type selection
        rnn_types = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU
        }
        
        if self.cell_type not in rnn_types:
            raise ValueError(f"Invalid cell_type. Choose from {list(rnn_types.keys())}")
            
        self.rnn = rnn_types[cell_type](
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # Output layer
        fc_input_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)
        
    def forward(self, x):
        """Forward pass of the model."""
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size, 
            self.hidden_size
        ).to(x.device)
        
        # Forward pass through RNN
        if self.cell_type == 'lstm':
            c0 = torch.zeros_like(h0)
            rnn_out, _ = self.rnn(x, (h0, c0))
        else:
            rnn_out, _ = self.rnn(x, h0)
            
        # Apply fully connected layer and squeeze output
        return self.fc(rnn_out).squeeze(-1)

# training.py
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List, Optional

class WaveTrainer:
    """Handles training and evaluation of wave prediction models."""
    
    def __init__(self, 
                 model: nn.Module, 
                 criterion=nn.MSELoss(),
                 learning_rate: float = 0.001,
                 milestones: Optional[List[int]] = None,
                 gamma: float = 0.1):
        """
        Args:
            model: The neural network model to train
            criterion: Loss function
            learning_rate: Initial learning rate
            milestones: List of epochs at which to decay the learning rate
            gamma: Multiplicative factor of learning rate decay
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Default milestones if none provided
        if milestones is None:
            milestones = [15, 30, 40]
            
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=gamma
        )
        
        # Store learning rate schedule for logging
        self.lr_history = []
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        
        for X, y in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            
        return running_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluate the model."""
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        for X, y in dataloader:
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            running_loss += loss.item()
            predictions.append(outputs)
            targets.append(y)
            
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        return running_loss / len(dataloader), predictions, targets
    
    def train(self, train_loader, val_loader, epochs: int = 50) -> Dict:
        """Train the model for specified number of epochs.
        
        Returns:
            Dict containing training history:
                - train_loss: List of training losses
                - val_loss: List of validation losses
                - val_mae: List of validation MAE scores
                - learning_rates: List of learning rates used
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            val_loss, y_pred, y_true = self.evaluate(val_loader)
            mae = nn.L1Loss()(y_pred, y_true).item()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(mae)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, mae={mae:.4f}, "
                  f"lr={current_lr:.6f}")
            
            # Step the scheduler
            self.scheduler.step()
            
        return history

# visualization.py
import matplotlib.pyplot as plt

def plot_learning_curves(history: Dict):
    """Plot training history including losses and learning rate.
    
    Args:
        history: Dictionary containing training history
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    ax2.plot(history['learning_rates'])
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    plt.tight_layout()


def plot_predictions(models_results: Dict, true_values: torch.Tensor):
    """Plot predictions from multiple models against true values."""
    # Individual model plots
    fig, axes = plt.subplots(len(models_results), 1, figsize=(15, 5*len(models_results)))
    
    for idx, (model_name, predictions) in enumerate(models_results.items()):
        ax = axes[idx] if len(models_results) > 1 else axes
        ax.plot(predictions.cpu().numpy().flatten(), 
                label=f'Predicted ({model_name})')
        ax.plot(true_values.cpu().numpy().flatten(), 
                label='True Values', linestyle='--')
        ax.set_title(f'{model_name} Predictions')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # Combined plot
    plt.figure(figsize=(15, 6))
    colors = ['r', 'g', 'b']
    
    for (model_name, predictions), color in zip(models_results.items(), colors):
        plt.plot(predictions.cpu().numpy().flatten(), 
                label=f'Predicted ({model_name})', 
                color=color, alpha=0.7)
    
    plt.plot(true_values.cpu().numpy().flatten(), 
            label='True Values', 
            color='k', linestyle='--')
    plt.title('All Models Predictions vs True Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate schedule
    if hasattr(models_results, 'learning_rates'):
        plt.figure(figsize=(10, 4))
        plt.plot(models_results['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    
    plt.show()
    
def plot_single_prediction(models_results: Dict, true_values: torch.Tensor, sample_idx: int = 0):
    """Plot predictions from multiple models against true values for a single sample.
    
    Args:
        models_results: Dictionary of model predictions
        true_values: True target values
        sample_idx: Index of the sample to plot (default: 0)
    """
    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b']
    
    # Get the sequence length from the true values
    seq_length = true_values.shape[1]
    x_axis = np.arange(seq_length)
    
    # Plot predictions from each model
    for (model_name, predictions), color in zip(models_results.items(), colors):
        single_prediction = predictions[sample_idx].cpu().numpy()
        plt.plot(x_axis, single_prediction, 
                label=f'Predicted ({model_name})', 
                color=color, 
                marker='o',
                markersize=6,
                alpha=0.7)
    
    # Plot true values
    single_true = true_values[sample_idx].cpu().numpy()
    plt.plot(x_axis, single_true, 
            label='True Values', 
            color='k', 
            linestyle='--',
            marker='s',
            markersize=6)
    
    plt.title('Model Predictions vs True Values (Single Sample)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Add value annotations
    for model_name, predictions in models_results.items():
        single_pred = predictions[sample_idx].cpu().numpy()
        for i, val in enumerate(single_pred):
            plt.annotate(f'{val:.2f}', 
                        (x_axis[i], single_pred[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    # Annotate true values
    for i, val in enumerate(single_true):
        plt.annotate(f'{val:.2f}', 
                    (x_axis[i], single_true[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=8)
    
    plt.tight_layout()


# main.py
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data()
    
    # Define learning rate schedule milestones
    milestones = [15, 30, 40]
    
    # Initialize models
    models = {
        'Simple RNN': WavePredictor(input_size=1, cell_type="rnn").to(device),
        'LSTM': WavePredictor(input_size=1, cell_type="lstm", 
                             bidirectional=True).to(device),
        'GRU': WavePredictor(input_size=1, cell_type="gru", 
                            bidirectional=True).to(device)
    }
    
    # Train and evaluate models
    results = {}
    histories = {}
    
    # Get a single batch from test loader for visualization
    test_batch = next(iter(test_loader))
    test_X, test_y = test_batch
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trainer = WaveTrainer(
            model,
            learning_rate=0.001,
            milestones=milestones,
            gamma=0.1
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        histories[name] = history
        
        # Get predictions for the single test batch
        with torch.no_grad():
            predictions = model(test_X)
        results[name] = predictions
        
        # Plot learning curves
        plot_learning_curves(history)
        plt.suptitle(f'{name} Training History')
        plt.show()
    
    # Plot predictions for a single sample
    sample_idx = 0  # You can change this to visualize different samples
    plot_single_prediction(results, test_y, sample_idx)
    plt.show()
    
    # Print error metrics for the sample
    print("\nError metrics for sample {}:".format(sample_idx))
    for name, predictions in results.items():
        single_pred = predictions[sample_idx]
        single_true = test_y[sample_idx]
        mse = nn.MSELoss()(single_pred, single_true).item()
        mae = nn.L1Loss()(single_pred, single_true).item()
        print(f"{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        
if __name__ == "__main__":
    main()