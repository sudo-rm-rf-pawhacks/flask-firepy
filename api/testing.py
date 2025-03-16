import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv("C:\\Users\\craze\\OneDrive\\Attachments\\Documents\\weather_data_3.csv")  # Replace with actual file path

X = df[['maxtempF', 'mintempF', 'avgtempF', 'totalSnow_cm', 'humid', 'wind', 'precip', 'sunHour', 'lat', 'long']]  # Adjust based on your dataset
y = df['had_wildfire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
print(f'Accuracy: {accuracy:.2f}')
print(f'ROC AUC: {roc_auc:.2f}')
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(importance_df)

def predict_wildfire(high_temp, low_temp, avg_temp, humidity, wind_speed, precip, sun_hour, lat, long):
    print("Enter weather data:")
    total_snow = 0
    user_data = np.array([[high_temp, low_temp, avg_temp, total_snow, humidity, wind_speed, precip, sun_hour, lat, long]])
    
    user_data = scaler.transform(user_data)
    
    probability = model.predict_proba(user_data)[:, 1][0]
    
    return(probability)

print(predict_wildfire(86.29032258, 46.64516129, 76.70967742, 32.70967742, 5.451612903, 0, 14.39677419, 39.58, -120.52))