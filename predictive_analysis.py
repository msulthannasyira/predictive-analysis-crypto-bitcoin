
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_and_preprocess_data(path, years=5):
    df = pd.read_csv(path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.rename(columns={'Open time': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    last_date = df.index[-1]
    start_date = last_date - pd.DateOffset(years=years)
    df = df[df.index >= start_date]

    return df

def plot_closing_price(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'])
    plt.title('Harga Penutupan Bitcoin - 5 Tahun Terakhir')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.show()

def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']])
    return scaled, scaler

def create_dataset(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'MSE : {mse:.2f}')
    print(f'MAE : {mae:.2f}')
    print(f'R² Score : {r2:.4f}')

def predict_future(model, input_seq, steps, scaler):
    predictions = []
    for _ in range(steps):
        next_value = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(next_value)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_value]]], axis=1)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

def main():
    df = load_and_preprocess_data('bitcoin_2018_2025.csv')
    plot_closing_price(df)

    scaled_data, scaler = scale_data(df)
    window_size = 1800
    X, y = create_dataset(scaled_data, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((window_size, 1))
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

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

    evaluate_model(y_test_inv, y_pred_inv)

    last_60_days = scaled_data[-window_size:]
    input_seq = last_60_days.reshape(1, window_size, 1)
    future_predictions = predict_future(model, input_seq, steps=30, scaler=scaler)

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)

    plt.figure(figsize=(12, 5))
    plt.plot(df.index[-200:], df['Close'].values[-200:], label='Data Historis (200 hari terakhir)')
    plt.plot(future_dates, future_predictions, label='Prediksi 30 Hari ke Depan', color='orange')
    plt.title('Prediksi Harga Bitcoin 30 Hari ke Depan (berdasarkan 5 Tahun Terakhir)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
