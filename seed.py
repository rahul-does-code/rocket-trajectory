from train import DataLoader, train_dataset, np, torch, SurrogateNet, criterion, mse, X_test, df_training, y_test 

def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

mlp_altitude_MSE = []
mlp_landing_MSE = []
mlp_altitude_predictions = []
mlp_landing_predictions = []

for seed in range(10):
    set_random(seed)
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
    model = SurrogateNet()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    num_epochs = 500

    for epoch in range(num_epochs):
        model.train() # puts in training mode
        for X_batch, y_batch in train_loader: # for small batch x_batch, y_batch is the target
            optimizer.zero_grad() # clears gradient
            outputs = model(X_batch) # forward pass by model to guess output based on param
            loss = criterion(outputs, y_batch) # how wrong were the ouputs compared to y_batch

            loss.backward() # decides what params need to be adjusted
            optimizer.step() # adjusts the parameters            

    model.eval() # put in eval mode

    with torch.no_grad(): # turns off pytorch gradients
        test_outputs = model(X_test) # runs fully-trained model on reserved X_test data

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

    mlp_altitude_MSE.append(mse(predicted_altitude, actual_altitude))
    mlp_landing_MSE.append(mse(predicted_landing, actual_landing))
    mlp_altitude_predictions.append(predicted_altitude)
    mlp_landing_predictions.append(predicted_landing)

mlp_altitude_predictions = np.array(mlp_altitude_predictions) # making it into an array
mlp_landing_predictions = np.array(mlp_landing_predictions) # making it into an array

print(f"MLP altitude: {np.mean(mlp_altitude_MSE):.2f} ± {np.std(mlp_altitude_MSE):.2f}")
print(f"MLP landing : {np.mean(mlp_landing_MSE):.2f} ± {np.std(mlp_landing_MSE):.2f}")

### creating the final plot ###

import matplotlib.pyplot as plt
from train import df, df_testing, linear_altitude, linear_landing, quadratic_altitude, quadratic_landing

altitude_mean = mlp_altitude_predictions.mean(axis = 0)
altitude_std  = mlp_altitude_predictions.std(axis = 0)
landing_mean = mlp_landing_predictions.mean(axis = 0)
landing_std  = mlp_landing_predictions.std(axis = 0)

test_angle = df_testing['angle'].values
all_angle  = df['angle'].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Altitude ---- #
ax1.plot(all_angle, df['max_altitude'].values, 'k-', label='True (sweep)')
ax1.axvspan(5, 75, alpha=0.1, color='gray', label='Training region')
ax1.plot(test_angle, altitude_mean, 'b-', label='MLP mean')
ax1.fill_between(test_angle, altitude_mean - altitude_std, altitude_mean + altitude_std, alpha=0.3, color='blue', label='MLP ±1σ')
ax1.plot(test_angle, linear_altitude, '--', label='Linear')
ax1.plot(test_angle, quadratic_altitude, '--', label='Quadratic')
ax1.set_xlabel('Angle (deg)'); ax1.set_ylabel('Max altitude (m)'); ax1.legend(); ax1.set_title('Altitude extrapolation')

# ---- Landing ---- #
ax2.plot(all_angle, df['landing_x'].values, 'k-', label='True (sweep)')
ax2.axvspan(5, 75, alpha=0.1, color='gray', label='Training region')
ax2.plot(test_angle, landing_mean, 'b-', label='MLP mean')
ax2.fill_between(test_angle, landing_mean - landing_std, landing_mean + landing_std, alpha=0.3, color='blue', label='MLP ±1σ')
ax2.plot(test_angle, linear_landing, '--', label='Linear')
ax2.plot(test_angle, quadratic_landing, '--', label='Quadratic')
ax2.set_xlabel('Angle (deg)'); ax2.set_ylabel('Landing distance (m)'); ax2.legend(); ax2.set_title('Landing extrapolation')

plt.tight_layout()
import time
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(fname = f"extrapolation_plot_{timestamp}", dpi=150)
plt.show()