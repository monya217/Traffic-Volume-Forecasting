import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading dataset...")
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# Preprocessing
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x>=5 else 0)
df['lag1'] = df['traffic_volume'].shift(1)
df['lag2'] = df['traffic_volume'].shift(2)
df = df.dropna().reset_index(drop=True)

feature_cols = [
    "hour","day","month","weekday","is_weekend",
    "temp","rain_1h","snow_1h","clouds_all",
    "lag1","lag2"
]

X = df[feature_cols]
y = df["traffic_volume"]

split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Training model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save to correct location
output_path = r"D:\College\Semester 7\RAI\lab5\models\traffic_rf_model.pkl"
joblib.dump(model, output_path)

print("Model saved successfully at:", output_path)
