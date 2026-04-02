from flask import Flask, request, jsonify, render_template
import pandas as pd
from utils.logger import log_data
from utils.predictor import predict
import threading
import time
import matplotlib.pyplot as plt

app = Flask(__name__)

DATA_FILE = "data/data_log.csv"

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

# 🔥 Receive data from ESP32
@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json
    log_data(data)
    return jsonify({"status": "saved"})

# 🔥 Send latest data to frontend
@app.route('/get-data')
def get_data():
    try:
        df = pd.read_csv(DATA_FILE)
        return df.tail(30).to_json(orient="records")
    except:
        return jsonify([])

# 🔥 Prediction API
@app.route('/predict')
def predict_route():
    return jsonify(predict())

def save_plots():
    while True:
        try:
            df = pd.read_csv("data/data_log.csv")

            plt.figure()
            plt.plot(df['distance'], label='Distance')
            plt.plot(df['brightness'], label='Brightness')
            plt.legend()
            plt.savefig("static/plots/latest.png")
            plt.close()

        except:
            pass

        time.sleep(10)


if __name__ == "__main__":
    # 🔥 Start background thread
    threading.Thread(target=save_plots, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)