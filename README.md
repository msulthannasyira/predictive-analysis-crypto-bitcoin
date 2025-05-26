# predictive-analysis-crypto-bitcoin
Analisis prediksi harga kripto bitcoin di masa depan

- Nama Lengkap: Muhammad Sulthan Nasyira
- Alur Belajar: Machine Learning Engineer
- Cohort ID: MC589D5Y2486
- Coding Camp Email Username:   mc589d5y2486@student.devacademy.id
- Email Terdaftar:   sulthanasyirah@gmail.com
- Group Belajar:   MC-49

## 1. Domain Proyek
### Latar belakang

Beberapa tahun terakhir, Bitcoin telah menjadi salah satu aset digital yang paling banyak diperbincangkan di dunia. Sebagai mata uang kripto pertama, Bitcoin tidak hanya dikenal sebagai alat tukar digital, tetapi juga telah berkembang menjadi instrumen investasi yang menarik perhatian banyak orang mulai dari investor individu hingga lembaga keuangan besar. Namun, satu hal yang menjadi ciri khas Bitcoin adalah harganya yang sangat fluktuatif dan sulit diprediksi.

Fluktuasi harga Bitcoin ini dipengaruhi oleh berbagai faktor, seperti sentimen pasar, regulasi pemerintah, teknologi blockchain yang terus berkembang, hingga isu-isu ekonomi global. Ketidakpastian ini tentu menjadi tantangan bagi para investor dan trader dalam mengambil keputusan yang tepat. Di sinilah peran teknologi, khususnya Machine Learning, menjadi sangat penting. Pendekatan ini mampu menganalisis data historis dan menemukan pola-pola tersembunyi yang mungkin luput dari metode analisis tradisional seperti analisis teknikal maupun fundamental.

### Mengapa Masalah Ini Penting dan Bagaimana Cara Menyelesaikannya
Masalah prediksi harga Bitcoin sangat relevan karena menyangkut keputusan finansial yang berisiko tinggi. Jika prediksi harga bisa dilakukan dengan lebih akurat, para investor bisa lebih percaya diri dalam membuat keputusan kapan harus membeli, menjual, atau menahan aset. Ini tidak hanya membantu mengurangi risiko kerugian, tapi juga meningkatkan efisiensi dan efektivitas dalam aktivitas jual beli aset kripto.

Melalui proyek ini, pendekatan Machine Learning digunakan untuk membangun model prediksi harga Bitcoin berdasarkan data historis, volume transaksi, dan indikator teknikal lainnya. Harapannya, model ini bisa menjadi alat bantu yang dapat memberikan prediksi harga jangka pendek atau menengah dengan tingkat akurasi yang lebih baik, sehingga memberikan gambaran tren yang lebih jelas bagi pengguna.

### Referensi
McNally, S., Roche, J., & Caton, S. (2018). Predicting the Price of Bitcoin Using Machine Learning. 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP). DOI: 10.1109/PDP2018.20180200044, penelitian ini membahas penggunaan RNN dan LSTM dalam memprediksi harga Bitcoin menggunakan data historis.

Investopedia. (n.d.). Bitcoin. Diakses dari: https://www.investopedia.com/terms/b/bitcoin.asp, laman ini menjelaskan secara lengkap apa itu Bitcoin, bagaimana cara kerjanya, dan apa saja yang memengaruhi nilainya.

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
Dataset ini berisi data historis harga Bitcoin per jam dari 1 Januari 2018 hingga 14 Mei 2025, dengan total 64.439 baris data dan 11 kolom. Dataset digunakan untuk membangun model prediksi harga Bitcoin menggunakan pendekatan LSTM (Long Short-Term Memory). Fokus utama ditujukan pada kolom Close (harga penutupan).

- Jumlah data (baris): 64,439 baris
- Jumlah kolom: 11 kolom
- Frekuensi data: Per jam (hourly)
- Rentang waktu data: 1 Januari 2018 sampai 14 Mei 2025
- Target variabel (label): `Close` (harga penutupan)

### Sumber Data
Dataset diperoleh dari Kaggle dengan tautan berikut: https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024?select=btc_1h_data_2018_to_2025.csv

### Deskripsi Variabel
- `Open time`: Waktu pembukaan candle (datetime), diubah menjadi indeks `Date`
- `Open`: Harga pembukaan
- `High`: Harga tertinggi
- `Low`: Harga terendah
- `Close`: Harga penutupan (variabel target)
- `Volume`: Jumlah Bitcoin yang diperdagangkan
- `Close time`: Waktu penutupan candle
- `Quote asset volume`: Volume dalam quote currency (USD)
- `Number of trades`: Jumlah transaksi
- `Taker buy base asset volume`: Volume beli oleh taker dalam satuan Bitcoin
- `Taker buy quote asset volume`: Volume beli oleh taker dalam satuan USD
- `Ignore`: Kolom yang tidak digunakan

Dari keseluruhan data, hanya kolom berikut yang digunakan untuk proses modeling prediksi harga Bitcoin:
- `Date` indeks waktu (dari `Open time`)
- `Close` (target variabel)

### Kualitas data
- Missing Values: Tidak ada missing values pada semua kolom
- Duplikasi data: Tidak ada baris duplikat terdeteksi (0 duplikasi)
- Outliner(indikasi awal): Ditemukan nilai maksimum (harga) yang jauh lebih tinggi dari nilai rata-rata, mengindikasikan adanya potensi outlier signifikan
- jumlah data yang bisa digunakan: 64,439 baris x 11 kolom

### Statistik Deskriptif (Kolom Harga)

| Statistik | Open     | High     | Low      | Close    |
|-----------|----------|----------|----------|----------|
| Mean      | 31,260.8 | 31,400.1 | 31,115.2 | 31,262.2 |
| Std       | 25,752.5 | 25,853.8 | 25,649.5 | 25,753.9 |
| Min       | 3,172.62 | 3,184.75 | 3,156.26 | 3,172.05 |
| 25%       | 9,184.17 | 9,219.47 | 9,149.84 | 9,184.40 |
| 50%       | 23,905.7 | 24,005.1 | 23,780.0 | 23,906.3 |
| 75%       | 46,859.2 | 47,117.2 | 46,582.3 | 46,859.6 |
| Max       | 108,320  | 109,588  | 107,781  | 108,320  |

Terlihat bahwa harga maksimum Bitcoin (108,320 USD) jauh di atas rata-rata (31,262 USD), mengindikasikan volatilitas yang sangat tinggi dan potensi outlier pada data harga.

### Eksplorasi dan Visualisasi Data (EDA)

Beberapa tahapan eksplorasi data yang dilakukan:

- Penyaringan Data, hanya data dari 5 tahun terakhir yang digunakan untuk meningkatkan relevansi prediksi.
- Sortir berdasarkan waktu untuk memastikan urutan kronologis data
- Visualisasi garis dari harga penutupan:

```python
plt.figure(figsize=(10, 5))
plt.plot(df['Close'])
plt.title('Harga Penutupan Bitcoin - 5 Tahun Terakhir')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.show()
```
Visualisasi tersebut memperlihatkan fluktuasi signifikan pada harga Bitcoin, termasuk tren bullish dan bearish, serta kemungkinan adanya pola musiman atau siklus pasar yang relevan untuk pemodelan LSTM.

## 4. Data Preparation

Tahap ini bertujuan untuk mempersiapkan data agar dapat digunakan secara optimal dalam proses pelatihan model prediksi harga Bitcoin berbasis LSTM. Proses data preparation dilakukan secara bertahap dengan mempertimbangkan karakteristik data time series yang digunakan. Karena model LSTM membutuhkan input dalam bentuk sekuensial, maka data harus memiliki format dan struktur tertentu agar model dapat memahami serta mempelajari pola pergerakan harga Bitcoin secara akurat.

### Pemilihan Rentang Waktu (Subsetting Data Historis)

Dataset yang digunakan berisi data historis harga Bitcoin dari tahun 2018 hingga 2025, dengan resolusi per jam. Namun, untuk fokus pada data yang paling relevan dengan kondisi pasar saat ini, hanya data 5 tahun terakhir yang digunakan. Pemilihan ini dilakukan dengan teknik time-based subsetting, dengan mengambil data dari `last_date - 5` tahun hingga `last_date`

```python
last_date = df.index[-1]
start_date = last_date - pd.DateOffset(years=5)
df = df[df.index >= start_date]
```

Alasan: Data lama berisiko memperkenalkan pola usang atau noise yang sudah tidak relevan dengan tren terbaru. Pemilihan rentang ini bersifat heuristik, menggabungkan pengetahuan domain (kondisi pasar berubah cepat) dan efisiensi dalam komputasi.

### Konversi Tipe Data dan Pengurutan Waktu

Sebelum dilakukan pemrosesan lebih lanjut, perlu dipastikan bahwa kolom indeks bertipe `datetime`. Setelah itu, data diurutkan secara kronologis dengan fungsi `sort_index()`:

```python
df.index = pd.to_datetime(df.index)
df = df.sort_index()
```

Urutan waktu sangat penting dalam time series. LSTM mempelajari hubungan antar waktu, sehingga data harus disusun secara berurutan.

### Penggantian Nama Kolom (Rename Columns)

Untuk konsistensi dan keterbacaan kode, dilakukan normalisasi nama kolom (contoh: kapitalisasi kolom close menjadi Close):

```python
df.rename(columns={'close': 'Close'}, inplace=True)
```

Langkah ini memastikan kolom yang digunakan memiliki nama yang sesuai dengan standar dalam kode

### Visualisasi Awal Data

Sebagai langkah eksploratif, dilakukan visualisasi harga penutupan dalam 5 tahun terakhir untuk mendapatkan gambaran umum tren harga.

```python
plt.figure(figsize=(10, 5))
plt.plot(df['Close'])
plt.title('Harga Penutupan Bitcoin - 5 Tahun Terakhir')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.show()
```

### Normalisasi Data

Karena model LSTM sensitif terhadap skala nilai input, dilakukan Min-Max Normalization menggunakan `MinMaxScaler` dari `sklearn`. Hanya kolom `Close` yang digunakan dalam model:

```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
```

Tujuannya agar normalisasi mempercepat konvergensi saat pelatihan dan mencegah dominasi nilai besar terhadap performa model.

###  Justifikasi Penggunaan Fitur Tunggal (Close)

Model ini hanya menggunakan satu fitur yaitu harga penutupan (`Close`) dengan alasan sebagai berikut:

- Harga penutupan merupakan indikator utama dalam banyak model prediktif karena mencerminkan konsensus akhir pasar
- Menghindari fitur seperti `Volume`, `High`, dan `Low` dilakukan untuk menjaga kesederhanaan model pada tahap awal
- Dalam pengembangan lebih lanjut, fitur tambahan bisa diuji secara eksperimen untuk melihat apakah mereka meningkatkan akurasi.

### Pembentukan Data Supervised (Sliding Window)

Data time series diubah menjadi format supervised learning menggunakan teknik sliding window. Input `X` terdiri dari 1800 jam terakhir (75 hari) untuk memprediksi harga pada jam berikutnya (`y`):

```python
def create_dataset(data, window_size=1800):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 1800
X, y = create_dataset(scaled_data, window_size)
```

- Nilai ini ditentukan secara heuristik berdasarkan asumsi bahwa 75 hari adalah periode yang cukup untuk model menangkap pola jangka menengah.
- Belum dilakukan hyperparameter tuning pada tahap ini, namun parameter ini dapat diuji dalam eksperimen selanjutnya untuk mencari nilai optimal.

### Transformasi Bentuk Data untuk LSTM
Model LSTM mengharuskan input dalam bentuk 3 dimensi: `[samples, timesteps, features]`. Oleh karena itu, data diubah bentuknya dengan `reshape()`

```python
X = X.reshape((X.shape[0], X.shape[1], 1))
```
- `samples`: jumlah window yang terbentuk dari dataset (jumlah sampel input).
- `timesteps`: panjang sekuens waktu = 1800 jam per sampel.
- `features`: hanya satu fitur, yaitu harga penutupan.

### Transformasi Bentuk Data untuk LSTM

Data dibagi menjadi dua bagian tanpa dilakukan shuffling:

- 80% untuk pelatihan (`X_train`, `y_train`)
- 20% untuk pengujian (`X_test`, `y_test`)

```python
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

- Rasio ini merupakan praktik umum dalam machine learning.
- Untuk time series, shuffling tidak dilakukan agar urutan waktu tetap terjaga.
- Pembagian ini memisahkan data lama (untuk pelatihan) dan data baru (untuk pengujian), yang mencerminkan situasi nyata dalam prediksi.

## 5. Modelling

Pada tahap ini, kita mulai membuat dan melatih model machine learning untuk memprediksi harga Bitcoin berdasarkan data historis, khususnya data harga penutupan. Model yang digunakan adalah Long Short-Term Memory (LSTM), yaitu salah satu jenis jaringan saraf tiruan yang dirancang khusus untuk memahami pola dalam data deret waktu (time series). LSTM sangat cocok digunakan dalam kasus seperti ini karena mampu mengingat pola data dari waktu ke waktu, sehingga dapat membantu memprediksi pergerakan harga di masa depan dengan lebih akurat.

### Pemilihan Model: Long Short-Term Memory (LSTM)

LSTM unggul dalam mengenali pola dalam data time series karena memiliki mekanisme gating yang mengatur informasi mana yang disimpan dan dilupakan. Hal ini membuat LSTM lebih tahan terhadap masalah vanishing gradient yang umum pada RNN biasa, dan lebih akurat dalam mengenali tren harga yang muncul dari waktu ke waktu.

### Arsitektur Model
Model dibangun menggunakan Keras dengan TensorFlow sebagai backend. Arsitektur model yang digunakan terdiri dari dua lapisan LSTM dan satu lapisan Dense output:

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
```
Penjelasan tahapan dan parameter:

- `LSTM(50, return_sequences=True)`: Lapisan LSTM pertama dengan 50 unit, dan mengembalikan output sequence penuh ke lapisan berikutnya
- `LSTM(50)`: Lapisan LSTM kedua dengan 50 unit yang hanya mengembalikan output terakhir
- `Dense(1)`: Layer output dengan satu neuron untuk memprediksi harga penutupan
- `optimizer='adam'`: Optimizer Adam digunakan karena adaptif dan umum untuk pelatihan deep learning
- `loss='mean_squared_error'`: Fungsi loss MSE cocok untuk regresi karena menghitung rata-rata kuadrat dari kesalahan prediksi.

Model dilatih dengan parameter:

```python
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
```
- `Epochs: 10` jumlah iterasi pelatihan
- `Batch Size: 32` jumlah sampel yang digunakan dalam satu iterasi update bobot.

### Evaluasi Model
Setelah pelatihan, model dievaluasi menggunakan data uji. Hasil prediksi dibandingkan dengan nilai aktual, kemudian diukur menggunakan beberapa metrik evaluasi regresi.
```python
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
```

Visualisasi hasil:
```python
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Prediksi Harga Bitcoin (Data Uji)')
plt.xlabel('Waktu')
plt.ylabel('Harga')
plt.legend()
plt.show()
```

Evaluasi metrik:
```python
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
```

Hasil evaluasi:
- MSE (Mean Squared Error): Mengukur rata-rata kesalahan kuadrat antara nilai aktual dan prediksi.
- MAE (Mean Absolute Error): Rata-rata selisih absolut antara nilai aktual dan prediksi.
- R² Score: Mengukur seberapa baik model menjelaskan variasi dalam data. Nilai mendekati 1 menandakan model yang baik.

### Kelebihan dan Kekurangan LSTM
Kelebihan:
- Mampu mengenali pola jangka panjang dalam data sekuensial.
- Efektif dalam memodelkan data time series seperti harga pasar.
- Mengatasi masalah vanishing gradient pada RNN biasa.
- Cocok untuk memprediksi data dengan tren dan pola musiman.

Kekurangan:
- Proses pelatihan lebih lambat dibanding model konvensional seperti regresi linier.
- Membutuhkan banyak data agar generalisasi model baik.
- Lebih kompleks dan membutuhkan tuning parameter seperti jumlah unit, batch size, dan epochs.
- Tidak selalu lebih baik dari model sederhana jika data tidak memiliki pola kompleks.

## 6. Evaluasi
Pada tahap ini, model LSTM yang sudah dilatih dievaluasi untuk melihat seberapa baik kemampuannya dalam memprediksi harga Bitcoin berdasarkan data historis. Evaluasi ini dilakukan dengan menggunakan beberapa metrik regresi yang memang cocok untuk jenis data deret waktu dan masalah prediksi nilai kontinu seperti harga. Tujuannya adalah untuk mengetahui apakah model sudah cukup akurat atau masih perlu diperbaiki agar hasil prediksinya bisa lebih mendekati kenyataan.

### Metrik Evaluasi yang Digunakan
Model dievaluasi menggunakan empat metrik utama yang umum pada masalah regresi:

#### Mean Absolute Error (MAE)
AE mengukur rata-rata selisih absolut antara nilai aktual dan nilai prediksi
- Kelebihannya Mudah diinterpretasikan sehingga memberikan ukuran kesalahan dalam satuan asli
- Kekurangannya tidak memberikan penalti lebih besar untuk kesalahan besar

#### Mean Squared Error (MSE)
MSE mengukur rata-rata selisih kuadrat antara nilai aktual dan nilai prediksi
- Kelebihannya mmberikan penalti lebih besar terhadap kesalahan besar
- Kekurangannya nilai MSE tidak berada pada skala asli data (karena dikuadratkan)

#### R² Score (Coefficient of Determination)
R² Score mengukur seberapa baik variabel independen menjelaskan variabel dependen. Nilai R² berkisar dari 0 sampai 1, di mana semakin dekat ke 1 menunjukkan model semakin baik.
- Kelebihannya menggambarkan proporsi variansi yang bisa dijelaskan oleh model
- Kekurangannya tidak selalu informatif jika target memiliki variansi rendah atau outlier tinggi

### Interpretasi Hasil
### MSE = 300,644.00
Mean Squared Error (MSE) menunjukkan rata-rata kuadrat dari kesalahan prediksi. Nilai 300,644 ini tergolong sangat kecil jika dibandingkan dengan harga Bitcoin yang bisa mencapai puluhan ribu hingga ratusan ribu USD. Nilai MSE yang rendah menunjukkan bahwa model mampu meminimalkan kesalahan prediksi secara efektif.

### MAE = 388.28
Rata-rata kesalahan prediksi model sekitar $388, yang relatif sangat kecil jika dibandingkan dengan harga Bitcoin yang biasanya berada dalam kisaran puluhan ribu USD. Ini menunjukkan model LSTM memiliki akurasi yang sangat baik dalam memprediksi harga Bitcoin berdasarkan data historis.

### R² Score = 0.9988
Nilai R² mendekati 1 (0.9988) menunjukkan bahwa 99.88% variasi dalam data harga Bitcoin dapat dijelaskan oleh model LSTM. Ini merupakan hasil yang sangat baik dan menunjukkan model mampu menangkap pola-pola penting dalam data historis, serta memiliki performa yang sangat andal untuk keperluan prediksi harga Bitcoin.

Domain proyek menyoroti pentingnya memanfaatkan Machine Learning dalam menangani volatilitas harga Bitcoin yang tinggi dan sulit diprediksi. Dengan hasil evaluasi di atas:
- Model LSTM terbukti efektif dalam memahami pola historis harga Bitcoin
- Kesalahan prediksi yang sangat kecil dan nilai R² yang mendekati 1 menunjukkan bahwa model mampu memberikan prediksi yang sangat mendekati kenyataan
- Hal ini sejalan dengan tujuan utama domain proyek, yaitu menggunakan teknologi Machine Learning untuk memetakan pola harga yang sulit ditangkap oleh metode tradisional.

### Kesimpulan

Volatilitas tinggi dan ketidakpastian harga Bitcoin menyebabkan kesulitan dalam pengambilan keputusan. Model prediktif yang dibangun terbukti dapat memberikan estimasi harga yang stabil dan akurat, membantu mengurangi ketidakpastian tersebut.

Tujuan membangun model yang dapat memprediksi harga jangka pendek dengan akurat telah dicapai:
- Visualisasi tren yang konsisten dan realistis
- Metrik evaluasi menunjukkan performa yang sangat baik

Solusi utama berupa penggunaan algoritma LSTM untuk prediksi time series terbukti efektif. Model tidak hanya memberikan prediksi numerik, tetapi juga visualisasi tren yang membantu pengambilan keputusan. Dampaknya sangat relevan bagi investor, trader, maupun analis.









