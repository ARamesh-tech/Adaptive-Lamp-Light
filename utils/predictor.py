import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

rf = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

SEQ_LEN = 5

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

lstm = LSTMModel()
lstm.load_state_dict(torch.load("models/lstm_model.pt"))
lstm.eval()

def predict():
    df = pd.read_csv("data/data_log.csv")

    if len(df) < SEQ_LEN:
        return {"text": "Not enough data", "score": 0}

    df['dist_change'] = df['distance'].diff().fillna(0).abs()

    # RF
    rf_input = df.tail(1)[['distance','brightness','dist_change']]
    rf_prob = rf.predict_proba(rf_input)[0][1]

    # LSTM
    features = df[['distance','brightness','dist_change']]
    scaled = scaler.transform(features)

    seq = scaled[-SEQ_LEN:]
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        lstm_prob = lstm(seq).item()

    # 🔥 Combined score (0–100)
    score = int((rf_prob * 0.5 + lstm_prob * 0.5) * 100)

    # Risk levels
    if score > 70:
        text = "⚠️ High Eye Strain Risk"
    elif score > 40:
        text = "⚡ Moderate Risk"
    else:
        text = "✅ Safe"

    return {"text": text, "score": score}