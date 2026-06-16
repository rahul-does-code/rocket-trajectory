import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Load data
df = pd.read_csv("data/sweep_results.csv")

# 2. Normalize inputs and outputs
df_training = df.iloc[:40]
df_testing = df.iloc[40:]

# 3. Train/test split
train_normalized = (df_training - df_training.min())/(df_training.max() - df_training.min())
test_normalized = (df_testing - df_training.min())/(df_training.max() - df_training.min())

# 4. Define net
X_train = torch.tensor(train_normalized[['angle']].values, dtype=torch.float32)
y_train = torch.tensor(train_normalized[['max_altitude', 'landing_x']].values, dtype=torch.float32)

X_test = torch.tensor(test_normalized[['angle']].values, dtype=torch.float32)
y_test = torch.tensor(test_normalized[['max_altitude', 'landing_x']].values, dtype=torch.float32)

# 5. Train
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

class SurrogateNet(nn.Module):
    def __init__(self):
        super(SurrogateNet, self).__init__()
        self.in_to_h1 = nn.Linear(in_features = 1, out_features = 64)
        self.h1_to_h2 = nn.Linear(in_features = 64, out_features = 64)
        self.h2_to_out = nn.Linear(in_features = 64, out_features = 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.in_to_h1(x)
        x = self.relu(x)
        x = self.h1_to_h2(x)
        x = self.relu(x)
        x = self.h2_to_out(x)
        return x

model = SurrogateNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 500

for epoch in range(num_epochs):
    model.train() # puts in training mode
    for X_batch, y_batch in train_loader: # for small batch x_batch, y_batch is the target
        optimizer.zero_grad() # clears gradient
        outputs = model(X_batch) # forward pass by model to guess output based on param
        loss = criterion(outputs, y_batch) # how wrong were the ouputs compared to y_batch

        loss.backward() # what param need to be adjusted?
        optimizer.step() # adjusts the parameters
    
    if epoch % 50 == 0: # after ever 50 epochs, print the loss
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                
# 6. Evaluate on test set
model.eval() # put in eval mode
test_loss = 0.0 # loss = 0

with torch.no_grad(): # turns off pytorch gradients
    test_outputs = model(X_test) # runs fully-trained model on reserved X_test data
    test_loss = criterion(test_outputs, y_test) # find error from predicted vs actual
    print(f"Test MSE: {test_loss.item():.6f}")

def plot_training():
    import matplotlib.pyplot as plt
    
    altitude_min = df_training['max_altitude'].min()
    altitude_max = df_training['max_altitude'].max()
    landing_min = df_training['landing_x'].min()
    landing_max = df_training['landing_x'].max()

    actual_altitude = y_test.numpy()[:,0] * (altitude_max - altitude_min) + altitude_min
    actual_landing = y_test.numpy()[:,1] * (landing_max - landing_min) + landing_min
    predicted_altitude = test_outputs.numpy()[:,0] * (altitude_max - altitude_min) + altitude_min
    predicted_landing = test_outputs.numpy()[:,1] * (landing_max - landing_min) + landing_min
    

    plt.subplot(121)
    diagonal = np.linspace(actual_altitude.min(), actual_altitude.max(), 100)
    plt.scatter(actual_altitude, predicted_altitude)
    plt.xlabel("Actual Max Altitude")
    plt.ylabel("Predicted Max Altitude")
    plt.plot(diagonal, diagonal, 'k--', label = "perfect")
    plt.legend()

    plt.subplot(122)
    diagonal = np.linspace(actual_landing.min(), actual_landing.max(), 100)
    plt.scatter(actual_landing, predicted_landing)
    plt.xlabel("Actual Landing")
    plt.ylabel("Predicted Landing")
    plt.plot(diagonal, diagonal, 'k--', label = "perfect")
    plt.legend()

    plt.show()

plot_training()