# 🌍 AQI Prediction Using Weather Data

This project predicts Air Quality Index (AQI) using machine learning based on weather parameters such as temperature, humidity, and pollutant levels. The model is optimized using a Random Forest Regressor to achieve high accuracy.

## 📌 Features
✅ Data Preprocessing & Cleaning  
✅ Feature Engineering & Outlier Handling  
✅ Machine Learning Model Training & Hyperparameter Tuning  
✅ Model Deployment on Google Cloud  
✅ Visualization of AQI Trends  

## 📂 Dataset
- The dataset includes weather and air pollution data from multiple locations.
- Features: `temperature`, `humidity`, `wind_speed`, `pm2.5`, `pm10`, `so2`, `no2`, etc.
- Target Variable: `AQI`

## 🚀 How to Run the Project
1️⃣ Clone the repository:  
```sh
git clone https://github.com/yourusername/AQI-Prediction-Using-Weather-Data.git
cd AQI-Prediction-Using-Weather-Data
pip install -r requirements.txt
python train_model.py
python app.py
 Access the API at: http://127.0.0.1:5000
📊 Model Performance
Accuracy: 99.21%
Precision: 99.23%
Recall: 99.21%
F1-score: 99.22%
MSE: 1.43 (Before Tuning), 1.67 (After Tuning)

