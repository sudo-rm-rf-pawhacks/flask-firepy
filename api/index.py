#from flask import Flask
#import requests
#import datetime
#import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os.path
print("reading csv")

df = pd.read_parquet(os.path.join(dirname, "data\\export_1742147442033.parquet"))
print("read csv")
X = df[['maxtempF', 'mintempF', 'avgtempF', 'totalSnow_cm', 'humid', 'wind', 'precip', 'sunHour', 'lat', 'long']]  # Adjust based on your dataset
y = df['had_wildfire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("past train test split")
scaler = StandardScaler()
#i know that these values are unused - the scaler method calls are necessary for the rest of the code to happen. the returned values are useful for testing. which i'm not doing currently, but i have done it in the past and may do it in the future, so for now these vars stay as they are.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("past x test")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("past fit")
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print('past importance df')
def predict_wildfire(data):
    user_data = np.array([[data["max_temp"], data["min_temp"], data["avg_temp"], 0, data["humidity"], data["wind_speed"], data["precipitation"], data["sun_hours"], data["lat"], data["long"]]])
    
    user_data = scaler.transform(user_data)
    
    probability = model.predict_proba(user_data)[:, 1][0]
    
    return(probability)


#Geocoders isn't necessary so long as we input GPS coordinates. If we input county, we do need geocoders to convert county to coords. If we ever switch back to county as input, just uncomment the commented lines
#from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request 
print('so it goes')
app = Flask(__name__) 
#geo-locator = Nominatim(user_agent = "fireguard")
key = "yU9qHcarLte1Fu65nMmrjwOp5BK38tS9"

@app.route("/test", methods = ['GET'])
def homepage():
    print('agsdfsdf')
    return("<p>smoke/p>")

@app.route('/api/<latitude>/<longitude>', methods = ['GET']) 
def get_weather_data(latitude, longitude):
    #location = geolocator.geocode(county)
    #longitude = location.longitude
    #latitude = location.latitude
    api_url = f"https://api.tomorrow.io/v4/weather/forecast?location={latitude},{longitude}&apikey={key}"
    raw_data = requests.get(api_url).json()
    #API returns a lot of data we don't need, so now I gotta filter it all out, then make it all usable
    days = []
    daily = raw_data["timelines"]["daily"]
    for i in range(6):
        daily_data = daily[i]
        values = daily_data["values"]

        sunrise_times = values["sunriseTime"].split("T")[1][:-1].split(":")
        sunrise_time = datetime.datetime(1970, 1, 1, int(sunrise_times[0]), int(sunrise_times[1]), int(sunrise_times[2]))

        sunset_times = values["sunsetTime"].split("T")[1][:-1].split(":")
        sunset_time = datetime.datetime(1970, 1, 1, int(sunset_times[0]), int(sunset_times[1]), int(sunset_times[2]))

        sun_time = abs(sunset_time - sunrise_time)
        sun_time = sun_time.total_seconds()/3600.0


        dict_to_add = {
            "date": daily_data["time"][:10],
            "max_temp": values["temperatureApparentMax"],
            "min_temp": values["temperatureApparentMin"],
            "avg_temp": values["temperatureApparentAvg"],
            "humidity": values["humidityAvg"],
            "wind_speed": values["windSpeedAvg"],
            "precipitation": values["precipitationProbabilityAvg"],
            "sun_hours": sun_time,
            "lat": latitude,
            "long": longitude
        }
        days.append(dict_to_add)

    probabilities = []
    for day in days:
        probabilities.append(predict_wildfire(day))
    return probabilities

if __name__ == '__main__': 
    app.run(host='0.0.0.0') 