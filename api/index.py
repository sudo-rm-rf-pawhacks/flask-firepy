from flask import Flask
import requests
import datetime
#Geocoders isn't necessary so long as we input GPS coordinates. If we input county, we do need geocoders to convert county to coords
#from geopy.geocoders import Nominatim
from flask import Flask, jsonify, request 
print('so it goes')
app = Flask(__name__) 
#geo-locator = Nominatim(user_agent = "fireguard")
key = "yU9qHcarLte1Fu65nMmrjwOp5BK38tS9"

@app.route("/test", methods = ['GET', 'POST'])
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
    #API returns a lot of data we don't need, so now I gotta filter it all out
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
            "sun_hours": sun_time
        }
        days.append(dict_to_add)

    return days

if __name__ == '__main__': 
    app.run(host='0.0.0.0') 