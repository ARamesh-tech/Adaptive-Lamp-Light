import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/data_log.csv")

# 🔥 Feature Engineering
df['dist_change'] = df['distance'].diff().fillna(0).abs()

# 🔥 IMPROVED LABELS (better learning)
df['strain'] = (
    ((df['distance'] < 1500) & (df['brightness'] > 200)) |
    (df['dist_change'] > 400)
).astype(int)

print("\nClass Distribution:")
print(df['strain'].value_counts())

# -------------------------
# 🌳 RANDOM FOREST
# -------------------------
X = df[['distance','brightness','dist_change']]
y = df['strain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n🔹 Random Forest Metrics")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, zero_division=1))
print("Recall:", recall_score(y_test, rf_pred, zero_division=1))
print("F1 Score:", f1_score(y_test, rf_pred, zero_division=1))

joblib.dump(rf, "models/rf_model.pkl")

# -------------------------
# 🔥 LSTM (PYTORCH)
# -------------------------

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

SEQ_LEN = 5

X_lstm, y_lstm = [], []

for i in range(len(scaled) - SEQ_LEN):
    X_lstm.append(scaled[i:i+SEQ_LEN])
    y_lstm.append(y.iloc[i+SEQ_LEN])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

# Convert to tensors
X_lstm = torch.tensor(X_lstm, dtype=torch.float32)
y_lstm = torch.tensor(y_lstm, dtype=torch.float32)

# 🔥 Shuffle data (important)
perm = torch.randperm(len(X_lstm))
X_lstm = X_lstm[perm]
y_lstm = y_lstm[perm]

# Train/Test split
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# -------------------------
# MODEL
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out  # 🔥 NO sigmoid here

lstm = LSTMModel()

# 🔥 Handle class imbalance
pos_weight = torch.tensor([(len(y_train_lstm) - y_train_lstm.sum()) / y_train_lstm.sum()])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

# -------------------------
# TRAINING
# -------------------------
for epoch in range(30):   # 🔥 increased epochs
    optimizer.zero_grad()
    logits = lstm(X_train_lstm).squeeze()
    loss = criterion(logits, y_train_lstm)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# -------------------------
# PREDICTION
# -------------------------
with torch.no_grad():
    logits = lstm(X_test_lstm).squeeze()
    probs = torch.sigmoid(logits)

    # 🔥 LOWER threshold (important)
    lstm_pred = (probs.numpy() > 0.4).astype(int)

print("\n🔹 LSTM (PyTorch) Metrics")
print("Accuracy:", accuracy_score(y_test_lstm, lstm_pred))
print("Precision:", precision_score(y_test_lstm, lstm_pred, zero_division=1))
print("Recall:", recall_score(y_test_lstm, lstm_pred, zero_division=1))
print("F1 Score:", f1_score(y_test_lstm, lstm_pred, zero_division=1))

# Save model
torch.save(lstm.state_dict(), "models/lstm_model.pt")