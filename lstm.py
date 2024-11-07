import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from parser import parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import time
import random

# Set the random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Detect if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directory to save checkpoints
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

output_dim = 10  # number of digits
# Hyperparameters
rnn_size = 64
num_layers = 2
bidirectional = True
dropout = 0.3
batch_size = 32
patience = 3
epochs = 25
lr = 1e-3
weight_decay = 1e-4


class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        if self.has_improved(value):
            self.patience_left = self.patience
            self.best = value
        else:
            self.patience_left -= 1

        return self.patience_left <= 0
    
    def is_best(self, value: float) -> bool:
        return self.has_improved(value)

    def has_improved(self, value: float) -> bool:
        return (self.best is None) or (
            value < self.best if self.mode == "min" else value > self.best
        )

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: List of numpy arrays, each of shape [seq_length x feature_dimension]
        labels: List of integer labels for each sequence
        """
        self.lengths = [len(i) for i in feats]
        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        padded = np.zeros((len(x), max(self.lengths), x[0].shape[1]))
        for i, seq in enumerate(x):
            padded[i, : len(seq)] = seq
        return padded

    def __getitem__(self, item):
        features = torch.tensor(self.feats[item], dtype=torch.float32)
        labels = torch.tensor(self.labels[item], dtype=torch.int64)
        length = torch.tensor(self.lengths[item], dtype=torch.int64)
        return features, labels, length

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=False,
        dropout=0.0,
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.rnn_size = rnn_size

        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """
        x : Tensor of shape [N x L x D]
        lengths: Tensor of shape [N]
        """
        # Sort sequences by length in descending order
        lengths, perm_idx = lengths.sort(0, descending=True)
        x = x[perm_idx]
        
        # Pack the sorted sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True
        )

        # Pass through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # For bidirectional LSTM, concatenate the final forward and backward hidden states
        if self.bidirectional:
            h_n = h_n.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            h_n = h_n[-1]

        # Apply dropout and fully connected layer
        out = self.dropout(h_n)
        out = self.fc(out)
        
        # Restore original sequence order
        _, unperm_idx = perm_idx.sort(0)
        out = out[unperm_idx]

        return out
        
        # Initialize hidden and cell states with zeros
        h_0 = torch.zeros(
            self.lstm.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.rnn_size,
            device=x.device
        )
        c_0 = torch.zeros(
            self.lstm.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.rnn_size,
            device=x.device
        )

        # LSTM forward pass
        out, _ = self.lstm(x, (h_0, c_0))

        # Get the outputs from the last timestep
        last_outputs = self.last_timestep(out, lengths, bidirectional=self.bidirectional)
        return self.fc(last_outputs)

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the sequence lengths
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx.to(outputs.device)).squeeze()


def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser(
        "./free-spoken-digit-dataset-1.0.10/recordings", n_mfcc=13
    )

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y, random_state=seed
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)

    # Initialize the training, val, and test dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False,
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        # Move data to the appropriate device
        features = features.to(device)
        labels = labels.to(device)
        #lengths = lengths.to(device)
        
        '''
        BONUS
        '''
        
        # lengths should remain on CPU for pack_padded_sequence
        lengths = lengths.cpu().long()

        # Sort sequences by lengths in descending order
        lengths, perm_idx = lengths.sort(0, descending=True)
        features = features[perm_idx]
        labels = labels[perm_idx]
        
        '''
        '''

        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        logits = model(features, lengths)
        # Compute loss
        loss = criterion(logits, labels)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch
            # Move data to the appropriate device
            features = features.to(device)
            labels = labels.to(device)
            #lengths = lengths.to(device)
            
            # lengths should remain on CPU for pack_padded_sequence
            lengths = lengths.cpu().long()
            
            # Sort sequences by length in descending order
            lengths, sort_idx = lengths.sort(0, descending=True)
            features = features[sort_idx]
            labels = labels[sort_idx]

            # Forward pass
            logits = model(features, lengths)
            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predictions
            outputs = torch.argmax(logits, dim=1)
            y_pred.extend(outputs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, np.array(y_pred), np.array(y_true)


def train(train_dataloader, val_dataloader, criterion):
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    model.to(device)  # Move model to the appropriate device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience, mode="min")
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    # Initialize y_pred and y_true for confusion matrix
    y_pred = torch.empty(0, dtype=torch.int8)
    y_true = torch.empty(0, dtype=torch.int8)
    # Start measuring time
    start_time = time.time()
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)
        valid_accuracy = accuracy_score(y_true, y_pred)
        print(
            f"Epoch {epoch + 1}: train loss = {training_loss:.4f}, valid loss = {valid_loss:.4f}, valid acc = {valid_accuracy:.4f}"
        )
        history['train_loss'].append(training_loss)
        history['val_loss'].append(valid_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Check if current model is the best
        if early_stopping.is_best(valid_loss):
            print("New best model found! Saving checkpoint.")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'epoch': epoch,
            }, best_model_path)
                    
        if early_stopping.stop(valid_loss):
            print("Early stopping...")
            break
        
    print(f"Training took: {time.time() - start_time:.2f} seconds")
    cm = confusion_matrix(y_pred, y_true, normalize='true')
    plot_confusion_matrix(cm, classes=np.arange(10), normalize=True, title='Validation Confusion Matrix Step 14')
    # Return the best model and the training history
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, history


# Create data loaders
train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)

# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()

# Train the model
model, history = train(train_dataloader, val_dataloader, criterion)

# Evaluate on the test set
test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
print(f"Test Loss: {test_loss:.4f}")
test_accuracy = accuracy_score(test_true, test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")


def plot_learning_curves(history):
    """Plot training history including losses and learning rate.
    
    Args:
        history: Dictionary containing training history
    """

    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/train_val_loss_step14.png')


# Plot the learning curves
plot_learning_curves(history)
plt.suptitle('Training History')
plt.show()

# Create the confusion matrices
test_cm = confusion_matrix(test_pred, test_true, normalize='true')
plot_confusion_matrix(test_cm, classes=np.arange(10), normalize=True, title='Test Confusion Matrix Step 14')

plt.tight_layout()
plt.show()      

# print classification report
from sklearn.metrics import classification_report

print(classification_report(test_true, test_pred, target_names=[str(i) for i in range(10)]))