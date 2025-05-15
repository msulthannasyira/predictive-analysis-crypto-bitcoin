# predictive-analysis-crypto-bitcoin
Analisis prediksi harga kripto bitcoin di masa depan

## 1. Domain Proyek
### Latar belakang

Dalam beberapa tahun terakhir, Bitcoin telah menjadi salah satu instrumen investasi dan spekulasi yang paling populer. Pergerakan harganya yang sangat fluktuatif sering kali membuat investor kesulitan dalam memprediksi arah tren di masa depan. Karena tingkat volatilitas yang tinggi ini, banyak orang mulai melirik pendekatan komputasi salah satunya Machine Learning sebagai cara untuk memprediksi harga Bitcoin, sebagai alternatif atau tambahan referensi dari metode analisis teknikal atau fundamental yang biasa digunakan.

### Mengapa masalah ini penting?
Masalah ini penting karena prediksi harga Bitcoin yang akurat bisa membantu investor membuat keputusan yang lebih tepat. Dengan begitu, mereka bisa meminimalkan risiko kerugian besar dan menjalankan aktivitas jual beli aset kripto dengan lebih efisien.

### Referensi
Beberapa studi telah dilakukan, seperti:
McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning.
Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
https://www.investopedia.com/terms/b/bitcoin.asp

## 2. Business Understanding

### Problem Statement
Harga Bitcoin sangat fluktuatif dan sulit diprediksi jika hanya mengandalkan metode tradisional. Karena itu, dibutuhkan pendekatan lain yang lebih adaptif, seperti pemanfaatan machine learning, untuk membantu memprediksi pergerakan harganya secara lebih akurat.

### Goal
Tujuan dari proyek ini adalah membangun model Machine Learning yang mampu memprediksi harga Bitcoin untuk hari berikutnya dengan akurat.

### Solution Statement
Solusi yang ditawarkan adalah mengembangkan model LSTM (Long Short-Term Memory) yang memanfaatkan data historis harga Bitcoin untuk mengenali pola-pola dari waktu ke waktu.

## 3. Data Undertanding

### Deskripsi Data
Dataset berisi data harga historis Bitcoin: tanggal, open, high, low, close, volume, dan market cap.
Jumlah data: ±3000 baris (lebih dari 500 sampel).
Format data harian dari tahun 2014–2024.

### Sumber Data

### Penjelasan Fitur
Date: Tanggal data direkam

Open: Harga pembukaan

High: Harga tertinggi

Low: Harga terendah

Close: Harga penutupan

Volume: Jumlah Bitcoin yang diperdagangkan

Market Cap: Total nilai pasar Bitcoin

### EDA

Plot tren harga Close dari waktu ke waktu

Korelasi antar fitur menggunakan heatmap

Distribusi volume perdagangan

## 4. Data Preparation

### Teknik yang digunakan

Handling missing values (drop/forward fill)

Normalisasi dengan Min-Max Scaler

Transformasi menjadi data time series (windowing)

Split data: train-test 80:20

Reshape data untuk input ke model LSTM

### Alasan Teknik

Time series memerlukan data dalam bentuk urutan

Normalisasi penting agar model tidak bias terhadap fitur berskala besar

Windowing agar model bisa belajar dari data historis sebelumnya

## 5. Modelling

### Model Baseline

Linear Regression menggunakan Open, High, Low, Volume sebagai fitur

### Model LSTM

2 LSTM layers, dropout, dan dense output layer

Optimizer: Adam

Loss function: MSE

### Pemilihan Model Terbaik

Model LSTM dipilih karena mampu menangkap pola waktu (temporal pattern) dengan lebih baik

Performa dibandingkan menggunakan RMSE dan MAE

Hyperparameter tuning dilakukan pada jumlah neuron dan window size

## 6. Evaluation

### Metrik Evaluasi

MSE (Mean Squared Error): Mengukur rata-rata error kuadrat

RMSE (Root Mean Squared Error): Akar dari MSE, lebih mudah diinterpretasikan karena satuannya sama dengan harga

### Hasil Evaluasi

Linear Regression: RMSE = 4500

LSTM: RMSE = 1500
LSTM memberikan error yang jauh lebih kecil, sehingga dipilih sebagai model terbaik.

## 7. Struktur Laporan

## Struktur Terorganisir
Mengikuti urutan: Domain → Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.

## Markdown & Gambar
Penjelasan setiap tahap ditulis dalam text cell dengan markdown. Visualisasi dimasukkan dengan matplotlib/seaborn. Format readable.











