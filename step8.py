import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

# Parameters
frequency = 40  # Hz
sampling_rate = 1000  # samples per second
duration = 2  # seconds
sequence_length = 10  # samples

# Generate time axis
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate sine and cosine waves
sine_wave = np.sin(2 * np.pi * frequency * t)
cosine_wave = np.cos(2 * np.pi * frequency * t)


# 2. Δημιουργία Ακολουθιών
X = []
y = []

for i in range(len(sine_wave) - sequence_length):
    X.append(sine_wave[i:i + sequence_length])
    y.append(cosine_wave[i + sequence_length])

X = np.array(X)
y = np.array(y)

# 3. Διαχωρισμός σε Training και Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 4. Μετατροπή σε PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train).float().unsqueeze(-1)  # [samples, seq_len, 1]
X_test_tensor = torch.from_numpy(X_test).float().unsqueeze(-1)
y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(-1)  # [samples, 1]
y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(-1)

'''
LSTM
'''

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Ορισμός LSTM στρώματος
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Ορισμός πλήρως συνδεδεμένου στρώματος
        # Εάν είναι bidirectional, το hidden_size διπλασιάζεται
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
    
    def forward(self, x):
        # Αρχικοποίηση των αρχικών κρυφών καταστάσεων
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)
        
        # Διέλευση μέσω LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Επιλογή του τελευταίου βήματος της ακολουθίας
        out = out[:, -1, :]
        
        # Διέλευση μέσω πλήρως συνδεδεμένου στρώματος
        out = self.fc(out)
        return out
    
    
'''
GRU
'''

class GRUModel(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Ορισμός GRU στρώματος
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Ορισμός πλήρως συνδεδεμένου στρώματος
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Διέλευση μέσω GRU
        out, _ = self.gru(x, h0)
        
        # Επιλογή του τελευταίου βήματος της ακολουθίας
        out = out[:, -1, :]
        
        # Διέλευση μέσω πλήρως συνδεδεμένου στρώματος
        out = self.fc(out)
        return out
    
class RNNModel(nn.Module):
    
    def __init__(self,input_size=1, hidden_size=50,num_layers=2,output_size=1):
        
        super(RNNModel,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        
        out,_ = self.rnn(x,h0)
        
        out = out[:,-1,:]
        
        out = self.fc(out)
        
        return out


'''
Training
'''

# 6. Επιλογή συσκευής
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 7. Δημιουργία μοντέλων
simple_rnn_model = RNNModel().to(device)
lstm_model = LSTMModel(bidirectional=True).to(device)  # Επιλογή Bidirectional
gru_model = GRUModel().to(device)

models = {
    'Simple RNN': simple_rnn_model,
    'LSTM': lstm_model,
    'GRU': gru_model
}

# 8. Ορισμός Optimizer και Loss Function
criterion = nn.MSELoss()

optimizers = { model_name: torch.optim.Adam(model.parameters(), lr=0.001) for model_name, model in models.items() }


# 9. Εκπαίδευση

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=16, shuffle=True
)

def training(model_name, model, train_loader, optimizer, criterion, device ,epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print (f'Model {model_name}: \n'
                       f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f'Model {model_name}: Average Epoch Loss: {avg_epoch_loss:.4f}')
        
# Start training
for model_name, model in models.items():
    training(model_name, model, train_loader, optimizers[model_name], criterion, device, epochs=100)
    
# 10. Αξιολόγηση Μοντέλων
def evaluate_model(model, X_test_tensor, y_test_tensor, device):
    model.eval()
    with torch.no_grad():
        inputs = X_test_tensor.to(device)
        targets = y_test_tensor.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()
        # Calculate Mean Absolute Error
        mae = nn.L1Loss()(outputs, targets).item()
    return loss, mae, outputs.cpu().numpy()

# Prepare for the plot of the true vs predicted values
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'orange']

# Plot the results of each ,model
for model_name, model in models.items():
    loss, mae, predictions = evaluate_model(model, X_test_tensor, y_test_tensor, device)
    print(f'Model {model_name}: Test Loss: {loss:.4f}, MAE: {mae:.4f}')
    plt.plot(predictions.flatten(), label=f'Προβλεπόμενο Συνημιτόνιο ({model_name})', color=colors.pop(0))
    
    
plt.plot(y_test_tensor.numpy().flatten(), label='Πραγματικό Συνημιτόνιο', color='blue')
plt.title('Πραγματικό vs Προβλεπόμενο Συνημιτόνιο')
plt.xlabel('Δείγματα')
plt.ylabel('Πλάτος')
plt.legend()
plt.grid(True)
plt.show()