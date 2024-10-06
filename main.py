import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('elnino/output.csv', na_values='.')
imputer = SimpleImputer(strategy='mean')
df[["s.s.temp.", "humidity", "zon.winds", "mer.winds", "air temp."]] = imputer.fit_transform(
    df[["s.s.temp.", "humidity", "zon.winds", "mer.winds", "air temp."]])

features = ['zon.winds', 'mer.winds', 'humidity', 'air temp.', 'latitude', 'longitude']
X = df[features]
y = df['s.s.temp.']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

y_test_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\nModel Evaluation:")
print(f"Neural Network MSE: {mse:.2f}, Rsquared: {r2:.2f}")

print("\nEnter the features for prediction:")
zon_winds = float(input("zonal Winds: "))
mer_winds = float(input("Meridion Winds: "))
humidity = float(input("Humidity: "))
air_temp = float(input("air Temperature: "))
latitude = float(input("Latitude: "))
longitude = float(input("longitude: "))

input_features = np.array([[zon_winds, mer_winds, humidity, air_temp, latitude, longitude]])

nn_prediction = model.predict(input_features)
print(f"\nPredicted s.s.t using NN: {nn_prediction[0][0]:.2f}")
