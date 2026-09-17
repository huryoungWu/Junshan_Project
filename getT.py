import pandas as pd
import requests

params = {
    "latitude": 30.5928,
    "longitude": 114.3055,
    "start_date": "2024-01-01",
    "end_date": "2025-12-31",
    "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m",
    "timezone": "Asia/Shanghai",
}

r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
r.raise_for_status()
j = r.json()

df = pd.DataFrame({
    "datetime": pd.to_datetime(j["hourly"]["time"]),
    "temperature_2m": j["hourly"]["temperature_2m"],
    "apparent_temperature": j["hourly"]["apparent_temperature"],
    "relative_humidity_2m": j["hourly"]["relative_humidity_2m"],
})

df.to_csv("wuhan_hourly_temperature_2024_2025.csv", index=False, encoding="utf-8-sig")