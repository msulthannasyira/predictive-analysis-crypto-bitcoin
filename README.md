# predictive-analysis-crypto-bitcoin
Analisis prediksi harga kripto bitcoin di masa depan

## 1. Domain Proyek
### Latar belakang

Beberapa tahun terakhir, Bitcoin telah menjadi salah satu aset digital yang paling banyak diperbincangkan di dunia. Sebagai mata uang kripto pertama, Bitcoin tidak hanya dikenal sebagai alat tukar digital, tetapi juga telah berkembang menjadi instrumen investasi yang menarik perhatian banyak orang—mulai dari investor individu hingga lembaga keuangan besar. Namun, satu hal yang menjadi ciri khas Bitcoin adalah harganya yang sangat fluktuatif dan sulit diprediksi.

Fluktuasi harga Bitcoin ini dipengaruhi oleh berbagai faktor, seperti sentimen pasar, regulasi pemerintah, teknologi blockchain yang terus berkembang, hingga isu-isu ekonomi global. Ketidakpastian ini tentu menjadi tantangan bagi para investor dan trader dalam mengambil keputusan yang tepat. Di sinilah peran teknologi, khususnya Machine Learning, menjadi sangat penting. Pendekatan ini mampu menganalisis data historis dan menemukan pola-pola tersembunyi yang mungkin luput dari metode analisis tradisional seperti analisis teknikal maupun fundamental.

### Mengapa Masalah Ini Penting dan Bagaimana Cara Menyelesaikannya
Masalah prediksi harga Bitcoin sangat relevan karena menyangkut keputusan finansial yang berisiko tinggi. Jika prediksi harga bisa dilakukan dengan lebih akurat, para investor bisa lebih percaya diri dalam membuat keputusan kapan harus membeli, menjual, atau menahan aset. Ini tidak hanya membantu mengurangi risiko kerugian, tapi juga meningkatkan efisiensi dan efektivitas dalam aktivitas jual beli aset kripto.

Melalui proyek ini, pendekatan Machine Learning digunakan untuk membangun model prediksi harga Bitcoin berdasarkan data historis, volume transaksi, dan indikator teknikal lainnya. Harapannya, model ini bisa menjadi alat bantu yang dapat memberikan prediksi harga jangka pendek atau menengah dengan tingkat akurasi yang lebih baik, sehingga memberikan gambaran tren yang lebih jelas bagi pengguna.

### Referensi
Beberapa studi telah dilakukan, seperti:
McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning.
Nakamoto, S. (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
https://www.investopedia.com/terms/b/bitcoin.asp

## 2. Business Understanding

### Problem Statement
Harga Bitcoin sangat fluktuatif dan sulit diprediksi jika hanya mengandalkan metode tradisional. Karena itu, dibutuhkan pendekatan lain yang lebih adaptif, seperti pemanfaatan machine learning, untuk membantu memprediksi pergerakan harganya secara lebih akurat.

### Goal
Tujuan dari proyek ini adalah membangun model Machine Learning yang mampu memprediksi harga Bitcoin untuk hari berikutnya dengan akurat. Metode ini bukan merupakan metode mutlak tetapi menjadi salah satu opsi 

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

### Struktur Terorganisir
Mengikuti urutan: Domain → Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation.

### Markdown & Gambar
Penjelasan setiap tahap ditulis dalam text cell dengan markdown. Visualisasi dimasukkan dengan matplotlib/seaborn. Format readable.











