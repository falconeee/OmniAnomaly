
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from omni_anomaly_torch import OmniAnomaly, preprocess_dataframe, TimeSeriesDataset, train_model, get_anomaly_scores
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data
# Sine wave + noise
t = np.linspace(0, 100, 2000)
data = np.sin(t) + np.random.normal(0, 0.1, 2000)
# Add anomaly
data[1500:1520] += 3.0

df = pd.DataFrame({'value': data, 'value2': np.cos(t) + np.random.normal(0, 0.1, 2000)})

# Split
train_size = 1000
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print("Data Shapes:", df_train.shape, df_test.shape)

# 2. Preprocess
window_size = 20
x_train, x_test, mean, std = preprocess_dataframe(df_train, df_test, window_size=window_size)
print("Windowed Shapes:", x_train.shape, x_test.shape)

# 3. Create DataLoaders
train_dataset = TimeSeriesDataset(x_train)
test_dataset = TimeSeriesDataset(x_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Initialize Model
x_dim = x_train.shape[2]
model = OmniAnomaly(x_dim=x_dim, z_dim=3, hidden_dim=32, window_length=window_size, nf_layers=2)

# 5. Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {device}...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_history = train_model(model, train_loader, optimizer, epochs=5, device=device)

# 6. Predict
scores = get_anomaly_scores(model, test_loader, device=device)
print("Score Shape:", scores.shape)

# Simple threshold
threshold = np.mean(scores) + 3 * np.std(scores)
anomalies = scores > threshold
print(f"Found {np.sum(anomalies)} anomalies.")

# 7. Visualization (Just print for now)
print("Top 5 Anomaly Scores:", np.sort(scores)[-5:])
