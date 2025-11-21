import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Load from LOCAL files (no URLs – no more 404 errors!)
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Prepare data
X_train = train_df.iloc[:, :-1].values.reshape(-1, 1, 187)
X_test = test_df.iloc[:, :-1].values.reshape(-1, 1, 187)
y_train = train_df.iloc[:, -1].values
y_test = test_df.iloc[:, -1].values

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model: 1D-ResNet style for ECG
class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 5)
        )
    def forward(self, x): return self.net(x)

model = ECGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 20 epochs
print("Training started...")
for epoch in range(1, 21):
    model.train()
    loss_sum = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    if epoch % 5 == 0 or epoch == 20:
        print(f"Epoch {epoch:02d} → loss: {loss_sum/len(train_loader):.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    pred = model(X_test).argmax(1).numpy()
    acc = accuracy_score(y_test, pred)
    print(f"\nFINAL ACCURACY: {acc*100:.2f}%")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['N','S','V','F','Q'], yticklabels=['N','S','V','F','Q'])
plt.title(f"MIT-BIH Arrhythmia Classification – {acc*100:.2f}% Accuracy")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix.png')  # Save as image
plt.show()

# Save model
torch.save(model.state_dict(), 'ecg_model.pth')
print("Model saved as ecg_model.pth – Project complete!")