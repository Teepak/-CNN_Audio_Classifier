import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim

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
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # Define the neural network architecture
    class Classifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # Initialize the model, loss function, and optimizer
    model = Classifier(X_train.shape[1], 256, len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001) # Add weight_decay parameter

    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # L2 regularization
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg # Add L2 regularization term to the loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 1 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        # Save the model
    PATH = 'models/modelAdam.pth'
    torch.save(model.state_dict(), PATH)

