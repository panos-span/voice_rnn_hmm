import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import math
from parser import parser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
bidirectional = True
dropout = 0.4
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.0


class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric hs not improved
        # Stop when patience reaches zero
        if self.has_improved(value):
            self.patience_left = self.patience
            self.best = value
        else:
            self.patience_left -= 1
        
        return self.patience_left <= 0

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        return (self.best is None) or (value < self.best if self.mode == "min" else value > self.best)


class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # TODO: YOUR CODE HERE
        self.lengths = [len(i) for i in feats]

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")
        

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        padded = np.zeros((len(x), max(self.lengths), x[0].shape[1]))
        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Zero pad the sequences and store them in the padded variable
        for i, seq in enumerate(x):
            padded[i, : len(seq)] = seq

        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

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

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=rnn_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_size, output_dim)
        

    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ , _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        last_outputs = self.last_timestep(out, lengths)
        return self.fc(self.dropout(last_outputs))

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()


def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser("./free-spoken-digit-dataset-1.0.10/recordings", n_mfcc=13)

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    # TODO: YOUR CODE HERE
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        # TODO: YOUR CODE HERE
        

        # zero grads in the optimizer
        optimizer.zero_grad()
        # run forward pass
        logits = model(features, lengths)
        
        # calculate loss
        loss = criterion(logits, labels)
        # TODO: YOUR CODE HERE
        # Run backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1
        print(f"Batch {num_batch} loss: {loss.item()}") # Print the loss every batch
        
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            # Run forward pass
            logits = model(features, lengths)
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            outputs = np.argmax(logits.cpu().numpy(), axis=1) # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(patience, mode="min")
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        # TODO: Calculate and print accuracy score
        valid_accuracy = accuracy_score(y_true, y_pred)
        print(
            "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                epoch, training_loss, valid_loss, valid_accuracy
            )
        )
        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break

    return model


train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = nn.CrossEntropyLoss()
input_dim = train_dataloader.dataset.feats.shape[-1]
model = BasicLSTM(input_dim, rnn_size, output_dim, num_layers, bidirectional=bidirectional, dropout=dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
model = train(train_dataloader, val_dataloader, criterion)

# TODO: YOUR CODE HERE
# print test loss and test accuracy
test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)
print(f"Test Loss: {test_loss}")
test_accuracy = accuracy_score(test_true, test_pred)
print(f"Test Accuracy: {test_accuracy}")
