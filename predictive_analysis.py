import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabulate import tabulate

# === 1. Load dan Pratinjau Data ===
df = pd.read_csv('bitcoin_2018_2025.csv')
df['Open time'] = pd.to_datetime(df['Open time'])
df = df.rename(columns={'Open time': 'Date'})
df.set_index('Date', inplace=True)
df = df.sort_index()

# Informasi dataset
print("\nInformasi Dataset:")
print(df.info())

# Cek missing values
missing_values = df.isnull().sum()
print("\nMissing Values per Kolom:")
print(tabulate(missing_values.reset_index().rename(columns={0: 'Missing Values', 'index': 'Kolom'}), headers='keys', tablefmt='fancy_grid'))

# Cek duplikasi
duplicate_count = df.duplicated().sum()
print(f"\nJumlah Duplikasi: {duplicate_count}")

# Statistik deskriptif
df_describe = df.describe()
print("\nStatistik Deskriptif:")
print(tabulate(df_describe, headers='keys', tablefmt='fancy_grid', showindex=True))

# === 2. Filter 5 Tahun Terakhir ===
last_date = df.index[-1]
start_date = last_date - pd.DateOffset(years=5)
df = df[df.index >= start_date]

# Visualisasi Harga Penutupan
plt.figure(figsize=(10, 5))
plt.plot(df['Close'])
plt.title('Harga Penutupan Bitcoin - 5 Tahun Terakhir')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.show()

# === 3. Normalisasi Data ===
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

# === 4. Membuat Dataset untuk LSTM ===
def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 1800
X, y = create_dataset(scaled_data, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data train dan test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === 5. Membangun Model LSTM ===
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# === 6. Evaluasi Model ===
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Prediksi Harga Bitcoin (Data Uji)')
plt.xlabel('Waktu')
plt.ylabel('Harga')
plt.legend()
plt.show()

mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
print(f'MSE : {mse:.2f}')
print(f'MAE : {mae:.2f}')
print(f'R² Score : {r2:.4f}')

# === 7. Prediksi 30 Hari ke Depan ===
last_60_days = scaled_data[-window_size:]
input_seq = last_60_days.reshape(1, window_size, 1)

future_predictions = []
for _ in range(30):
    next_pred = model.predict(input_seq, verbose=0)[0, 0]
    future_predictions.append(next_pred)
    input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

plt.figure(figsize=(12, 5))
plt.plot(df.index[-200:], df['Close'].values[-200:], label='Data Historis (200 hari terakhir)')
plt.plot(future_dates, future_predictions_inv, label='Prediksi 30 Hari ke Depan', color='orange')
plt.title('Prediksi Harga Bitcoin 30 Hari ke Depan (berdasarkan 5 Tahun Terakhir)')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.tight_layout()
plt.show()
