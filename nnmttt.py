import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
data = pd.read_csv('all_positions.csv')

# Step 2: Preprocess the data
def preprocess_data(data):
    # Filter out 'Undecided' positions
    data = data[data['winner'] != 'U']

    # Convert board positions to one-hot encoded format
    def one_hot_encode(position):
        encoded = []
        for cell in position:
            if cell == 'X':
                encoded += [1, 0, 0]
            elif cell == 'O':
                encoded += [0, 1, 0]
            else:
                encoded += [0, 0, 1]
        return encoded

    def winner_to_label(winner):
        if winner == 'X':
            return 0
        elif winner == 'O':
            return 1
        else:  # Tie
            return 2

    X = np.array([one_hot_encode(position) for position in data['position']])
    y = np.array([winner_to_label(winner) for winner in data['winner']])

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(data)

# Step 3: Create a neural network model: 27 x 64 x 32 x 3
class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        self.fc1 = nn.Linear(27, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = TicTacToeNN()

# Step 4: Train the model
class TicTacToeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = TicTacToeDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Step 5: Evaluate the model
test_dataset = TicTacToeDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')


def test_position(model, position):
    def one_hot_encode(position):
        encoded = []
        for cell in position:
            if cell == 'X':
                encoded += [1, 0, 0]
            elif cell == 'O':
                encoded += [0, 1, 0]
            else:
                encoded += [0, 0, 1]
        return encoded

    def label_to_winner(label):
        if label == 0:
            return 'X'
        elif label == 1:
            return 'O'
        else:  # Tie
            return 'T'

    encoded_position = one_hot_encode(position)
    input_tensor = torch.tensor([encoded_position], dtype=torch.float)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output.data, 1)
        winner = label_to_winner(predicted_label.item())

    return winner

position = "  O" \
           " X " \
           "OX "

predicted_winner = test_position(model, position)
print(f"Predicted winner for position:\n{position}\nWinner: {predicted_winner}")
