import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Load data
df = pd.read_csv("data/sweep_results.csv")

# 2. Train/test split on angles less than/greater than 75
df_training = df[df['angle'] < 75]
df_testing = df[df['angle'] >= 75]

# 3. Normalize inputs and outputs
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
    
criterion = nn.MSELoss()

def mse(predicted, actual):
    return np.mean((predicted - actual) ** 2)

# creating a linear baseline
train_angle = df_training['angle'].values
test_angle = df_testing['angle'].values

altitude_fit = np.poly1d(np.polyfit(train_angle, df_training['max_altitude'].values, 1))
landing_fit = np.poly1d(np.polyfit(train_angle, df_training['landing_x'].values, 1))
altitude_fit_quad = np.poly1d(np.polyfit(train_angle, df_training['max_altitude'].values, 2))
landing_fit_quad = np.poly1d(np.polyfit(train_angle, df_training['landing_x'].values, 2))

# running baseline
linear_altitude = altitude_fit(test_angle)
linear_landing = landing_fit(test_angle)
quadratic_altitude = altitude_fit_quad(test_angle)
quadratic_landing = landing_fit_quad(test_angle)

def plot_training():
    import matplotlib.pyplot as plt
        
    # plotting everything
    plt.subplot(121)
    diagonal = np.linspace(actual_altitude.min(), actual_altitude.max(), 100)
    # diagonal for reference line of what it should be
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

    import time
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(fname = f"my_plot_{timestamp}")
    plt.show()

if __name__ == "__main__":

    model = SurrogateNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 500

    for epoch in range(num_epochs):
        model.train() # puts in training mode
        for X_batch, y_batch in train_loader: # for small batch x_batch, y_batch is the target
            optimizer.zero_grad() # clears gradient
            outputs = model(X_batch) # forward pass by model to guess output based on param
            loss = criterion(outputs, y_batch) # how wrong were the ouputs compared to y_batch

            loss.backward() # decides what params need to be adjusted
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

    # easier normalization
    altitude_min = df_training['max_altitude'].min()
    altitude_max = df_training['max_altitude'].max()
    landing_min = df_training['landing_x'].min()
    landing_max = df_training['landing_x'].max()

    # normalizing + converting df into numpy array
    actual_altitude = y_test.numpy()[:,0] * (altitude_max - altitude_min) + altitude_min
    actual_landing = y_test.numpy()[:,1] * (landing_max - landing_min) + landing_min
    predicted_altitude = test_outputs.numpy()[:,0] * (altitude_max - altitude_min) + altitude_min
    predicted_landing = test_outputs.numpy()[:,1] * (landing_max - landing_min) + landing_min

    # NUMBERS ARE BASED ON ONE SINGLE RUN, NOT REPRESENTATIVE
    # USE SEED.PY FOR AVG ACROSS MULTIPLE SEEDS!!!
    print("=== Per-output test MSE (real units) ===")
    print(f"MLP    altitude: {mse(predicted_altitude, actual_altitude):.2f}")
    print(f"Linear altitude: {mse(linear_altitude,  actual_altitude):.2f}")
    print(f"Quad  altitude : {mse(quadratic_altitude, actual_altitude):.2f}") 
    print(f"MLP    landing : {mse(predicted_landing, actual_landing):.2f}")  
    print(f"Linear landing : {mse(linear_landing, actual_landing):.2f}") 
    print(f"Quad   landing : {mse(quadratic_landing, actual_landing):.2f}") 

    plot_training()