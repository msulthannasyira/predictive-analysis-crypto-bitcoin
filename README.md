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
McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning. 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP). DOI: 10.1109/PDP2018.20180200044, penelitian ini membahas penggunaan RNN dan LSTM dalam memprediksi harga Bitcoin menggunakan data historis.

Investopedia. (n.d.). Bitcoin. Diakses dari: https://www.investopedia.com/terms/b/bitcoin.asp, laman ini enjelaskan secara lengkap apa itu Bitcoin, bagaimana cara kerjanya, dan apa saja yang memengaruhi nilainya.

## 2. Business Understanding

### Problem Statements
Bitcoin adalah salah satu aset digital yang paling populer saat ini, namun juga dikenal memiliki harga yang sangat fluktuatif. Perubahan harganya bisa terjadi secara tiba-tiba, dipengaruhi oleh banyak faktor seperti berita global, kebijakan pemerintah, hingga sentimen publik. Hal ini membuat prediksi harga Bitcoin menjadi tantangan besar, terutama jika hanya mengandalkan metode analisis tradisional seperti analisis teknikal atau fundamental.

Ketidakpastian ini bisa merugikan investor baik yang berpengalaman maupun pemula karena keputusan beli atau jual menjadi kurang akurat. Oleh karena itu, diperlukan pendekatan yang lebih adaptif, seperti pemanfaatan teknologi Machine Learning, untuk membantu memprediksi pergerakan harga Bitcoin dengan lebih akurat.

### Goals
Tujuan dari proyek ini adalah membangun model prediksi harga Bitcoin berbasis Machine Learning yang dapat memperkirakan harga untuk hari berikutnya. Model ini diharapkan bisa menangkap pola dari data historis dan memberikan hasil yang lebih akurat dibandingkan dengan metode konvensional.

Model ini tidak bertujuan untuk menjadi satu-satunya acuan dalam pengambilan keputusan, tetapi sebagai alat bantu tambahan yang bisa memberikan pandangan berbasis data, terutama bagi mereka yang ingin mengurangi risiko dalam aktivitas jual beli aset kripto.

### Solution Statement
Untuk mencapai tujuan tersebut, proyek ini mengembangkan beberapa pendekatan yang berfokus pada pemrosesan data historis dan penggunaan algoritma deep learning yang sesuai dengan karakteristik data deret waktu (time series predictive analysis)

Model LSTM
Pendekatan utama adalah membangun model prediksi menggunakan algoritma Long Short-Term Memory (LSTM), yang merupakan jenis Recurrent Neural Network (RNN) dan sangat cocok untuk memproses data deret waktu seperti harga Bitcoin harian.

Validasi Model dengan Visualisasi dan Evaluasi
Selain menggunakan metrik kuantitatif, hasil model juga divalidasi secara visual dengan membandingkan prediksi terhadap data aktual pada data uji (testing set). Grafik ini membantu dalam memahami seberapa baik model mengikuti tren pergerakan harga secara umum.

## 3. Data Undertanding

### Informasi Umum Dataset
Dataset yang digunakan berisi data historis harga Bitcoin per jam dari tahun 2018 hingga awal 2025. Dataset ini digunakan untuk melakukan prediksi harga Bitcoin menggunakan model LSTM (Long Short-Term Memory). Fokus utama dalam proyek ini adalah memanfaatkan data 5 tahun terakhir sebagai basis prediksi.

- Jumlah total data awal: tergantung dari file `(bitcoin_2018_2025.csv)`, namun hanya subset 5 tahun terakhir yang digunakan.

- Frekuensi data: per jam

- Rentang data yang digunakan: ± awal 2020 sampai 2025

- Target variabel yang dianalisis dan diprediksi: Harga Penutupan (Close)

### Sumber Data
Dataset diperoleh dari Kaggle dengan tautan berikut: https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024?select=btc_1h_data_2018_to_2025.csv

### Deskripsi Variabel

`Open time`: Waktu pembukaan candle (format datetime)

`Open`: Harga pembukaan

`High`: Harga tertinggi

`Low`: Harga terendah

`Close`: Harga penutupan (variabel target)

`Volume`: Jumlah Bitcoin yang diperdagangkan

`Close time`: Waktu penutupan candle

`Quote asset volume`: Volume aset dalam quote currency

`Number of trades`: Jumlah transaksi

`Taker buy base asset volume`: Volume beli oleh taker dalam base asset

`Taker buy quote asset volume`: Volume beli oleh taker dalam quote asset

`Ignore`: Kolom yang tidak digunakan

Dalam alur yang sudah dibuat, hanya kolom `Open time` dan `Close` yang digunakan. Kolom `Open time` diubah menjadi `Date` dan dijadikan indeks waktu.



### Eksplorasi dan Visualisasi Data (EDA)

Beberapa tahapan eksplorasi data yang dilakukan:

- Penyaringan Data, hanya data dari 5 tahun terakhir yang digunakan untuk meningkatkan relevansi prediksi.

- Sortir berdasarkan waktu, data diurutkan berdasarkan indeks Date.

- Visualisasi Historis Harga Penutupan:

```python
plt.plot(df['Close'])
plt.title('Harga Penutupan Bitcoin - 5 Tahun Terakhir')
```
Visualisasi ini menunjukkan tren naik turun harga Bitcoin secara historis, mempermudah identifikasi fluktuasi dan pola musiman.


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











