# 💡 Adaptive Lamp Light System

## 📌 Overview

The Adaptive Lamp Light System is an intelligent IoT + Machine Learning
based solution designed to optimize lighting conditions based on user
distance and behavior to reduce eye strain and improve reading comfort.

------------------------------------------------------------------------

## ⚙️ System Components

### 🔹 ESP32 (IoT Layer)

-   Reads distance using IR sensor (analog input)
-   Controls RGB LED using PWM
-   Sends real-time data to Flask server via WiFi

### 🔹 Flask Backend

-   Receives sensor data
-   Stores data in CSV
-   Serves dashboard and ML predictions

### 🔹 Machine Learning Models

-   Random Forest (baseline)
-   LSTM (time-series learning using PyTorch)
-   Ensemble prediction system

------------------------------------------------------------------------

## 🏗️ Project Architecture

    ESP32 → Flask API → CSV Storage → ML Models → Dashboard Visualization

------------------------------------------------------------------------

## 🔄 Workflow

1.  ESP32 reads distance data
2.  Determines lighting profile (warm/neutral/cool)
3.  Sends data to Flask server
4.  Flask stores data in CSV
5.  ML models process data
6.  Dashboard visualizes real-time data + predictions

------------------------------------------------------------------------

## 🔁 Project Pipeline

    Sensor Data → Feature Engineering → ML Models → Prediction → Visualization → Alerts

------------------------------------------------------------------------

## 🧠 Machine Learning Models

### 🌳 Random Forest

-   Learns static relationships between:
    -   Distance
    -   Brightness
    -   Distance change
-   High accuracy due to rule-based labels

### 🔥 LSTM (PyTorch)

-   Learns temporal patterns (sequence behavior)
-   Captures user reading habits over time

------------------------------------------------------------------------

## 📊 Model Performance

### Dataset Distribution

-   Safe (0): 680
-   Risk (1): 106

### Random Forest Metrics

-   Accuracy: 1.0
-   Precision: 1.0
-   Recall: 1.0
-   F1 Score: 1.0

### LSTM Metrics

-   Accuracy: 0.159
-   Precision: 0.159
-   Recall: 1.0
-   F1 Score: 0.274

------------------------------------------------------------------------

## 📈 Dashboard Visualizations

### 1. Line Chart

-   X-axis: Timestamp
-   Y-axis: Distance & Brightness
-   Shows real-time trends

### 2. Bar Chart

-   Compares distance and brightness values

### 3. Pie Chart

-   Distribution of temperature (warm/neutral/cool)

### 4. Histogram

-   Distance distribution (Close/Medium/Far)

### 5. Bubble Chart

-   X-axis: Distance
-   Y-axis: Brightness
-   Bubble size: Data density

### 6. Gauge Meter

-   Displays eye strain score (0--100)
-   Visual risk indicator

------------------------------------------------------------------------

## 🎯 Eye Strain Prediction

-   Combines Random Forest + LSTM
-   Outputs:
    -   Safe
    -   Moderate Risk
    -   High Risk

------------------------------------------------------------------------

## ⚠️ Alerts

-   Triggered when eye strain score \> 50%
-   Warns about:
    -   Myopia
    -   Hypermetropia

------------------------------------------------------------------------

## 👁️ Use Cases

-   Personalized reading lamp
-   Student study environments
-   Office desk lighting
-   Eye strain prevention system

------------------------------------------------------------------------

## 🚀 Other Applications

-   Smart classrooms
-   Healthcare monitoring
-   Smart homes
-   AR/VR adaptive lighting

------------------------------------------------------------------------

## 🧠 Conclusion

This project integrates IoT, Machine Learning, and Web Technologies to
create a real-time adaptive lighting system that enhances user comfort
and reduces eye strain.

------------------------------------------------------------------------

## 👨‍💻 Author

A Ramesh
