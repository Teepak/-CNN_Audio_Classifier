import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import RMSprop


def train_model():
    # Load data
    data_path = 'dataset/mfcc_spec_features.csv'
    data = pd.read_csv(data_path)

    # Split data into features and labels
    X = np.array(data.iloc[:, 2:])
    y = data.iloc[:, 1]

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define the neural network architecture
    class MyModel(nn.Module):
        def __init__(self, input_size):
            super(MyModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc3(out)
            return out

    # Initialize the model, loss function, and optimizer
    model = MyModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = RMSprop(model.parameters(), lr=0.02)


    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch+1) % 10 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Evaluate the model on test data
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test.view(-1, 1))
    print('Test Loss: {:.4f}'.format(test_loss.item()))

    # Save the model
    PATH = 'models/modelRMSprop.pth'
    torch.save(model.state_dict(), PATH)
