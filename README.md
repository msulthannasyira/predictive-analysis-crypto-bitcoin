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

Beberapa tahun terakhir, Bitcoin telah menjadi salah satu aset digital yang paling banyak diperbincangkan di dunia. Sebagai mata uang kripto pertama, Bitcoin tidak hanya dikenal sebagai alat tukar digital, tetapi juga telah berkembang menjadi instrumen investasi yang menarik perhatian banyak orang—mulai dari investor individu hingga lembaga keuangan besar. Namun, satu hal yang menjadi ciri khas Bitcoin adalah harganya yang sangat fluktuatif dan sulit diprediksi.

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

Pada tahap ini, dilakukan berbagai langkah untuk mempersiapkan data agar siap digunakan dalam pelatihan model prediksi harga Bitcoin. Proses ini disebut data preparation, dan dilakukan secara bertahap serta disesuaikan dengan karakteristik data time series yang digunakan. Karena model yang dipakai adalah LSTM, maka data perlu disusun dengan cara tertentu agar bisa "dipahami" oleh model. Dengan kata lain, kita memastikan data memiliki format dan struktur yang tepat, sehingga model bisa belajar pola dari pergerakan harga Bitcoin secara akurat.

### Subsetting Data Historis (Pemilihan Rentang Waktu)

Teknik yang digunakan yaitu Time-based subsetting dengan mengunduh dataset yang berisi data historis harga Bitcoin dari tahun 2018 hingga 2025 (dalam resolusi 1 jam). Namun, untuk fokus pada data yang paling relevan dan mencerminkan tren pasar terkini, hanya data dari 5 tahun terakhir yang digunakan. Hal ini dilakukan dengan cara mengambil data dari `last_date - 5 tahun` hingga tanggal terakhir dalam dataset.

```python
last_date = df.index[-1]
start_date = last_date - pd.DateOffset(years=5)
df = df[df.index >= start_date]
```

Alasan dari pemilihan rentang waktu karena data yang lebih lama cenderung kurang mencerminkan kondisi pasar saat ini dan dapat memperburuk kinerja model karena memasukkan noise atau pola yang sudah tidak relevan.

### Pengurutan Data Berdasarkan Waktu

teknik yang digunakan yaitu Time index sorting dengan mengubah indeks data menjadi format tanggal kemudian diurutkan kembali berdasarkan waktu menggunakan fungsi `sort_index()`.

```python
df = df.sort_index()
```

Model LSTM mempelajari pola sekuensial dalam data. Oleh karena itu, urutan waktu harus konsisten agar model dapat menangkap dependensi temporal dengan benar.

### Normalisasi Data

Teknik yang digunakan dengan Min-Max Normalization sehingga Hanya kolom `Close` (harga penutupan) yang digunakan sebagai fitur untuk pelatihan model. Nilai-nilai pada kolom ini kemudian dinormalisasi ke dalam rentang [0,1] menggunakan `MinMaxScaler` dari `sklearn`.

```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])
```

Model deep learning seperti LSTM sensitif terhadap skala fitur. Normalisasi diperlukan agar proses pelatihan menjadi lebih stabil, cepat konvergen, dan tidak bias terhadap nilai yang besar.

### Pembuatan Window Data (Sliding Window / Time Series Framing)

Teknik yang digunakan dengan Sliding window with fixed window size dimana Data time series diubah menjadi format supervised learning, yaitu `X` (input) dan `y` (target output), menggunakan teknik sliding window. Dalam hal ini, setiap input terdiri dari 1800 jam (75 hari) harga penutupan sebelumnya untuk memprediksi harga jam berikutnya.

```python
def create_dataset(data, window_size=1800):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)
```

LSTM membutuhkan input dalam bentuk sekuens (urutan) agar dapat mempelajari hubungan antar waktu. Sliding window membantu memecah data menjadi pasangan input-output yang sesuai untuk pelatihan model prediktif.

### Transformasi Bentuk Data untuk LSTM

Teknik yang digunakan yaitu Reshaping to 3D structurem dimana LSTM memerlukan data input dalam bentuk tiga dimensi yaitu `[samples, timesteps, features]`. Oleh karena itu, array X diubah bentuknya menggunakan `reshape()`.

```python
X = X.reshape((X.shape[0], X.shape[1], 1))
```

Struktur ini diperlukan agar setiap contoh input terdiri dari `timesteps` urutan waktu, masing-masing dengan satu fitur, sesuai dengan spesifikasi arsitektur LSTM.

### Pembagian Data untuk Pelatihan dan Pengujian

Teknik yang digunakan yaitu Train-Test Split (80:20 ratio) dengna membagi Dataset menjadi dua bagian, 80% untuk pelatihan (`X_train, y_train`) dan 20% untuk pengujian (`X_test, y_test`).

```python
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
```

Pembagian ini diperlukan untuk melatih model pada sebagian besar data dan menguji performanya pada data yang belum pernah dilihat sebelumnya, sehingga dapat mengukur kemampuan generalisasi model.

## 5. Modelling

Pada tahap ini, kita mulai membuat dan melatih model machine learning untuk memprediksi harga Bitcoin berdasarkan data historis, khususnya data harga penutupan. Model yang digunakan adalah Long Short-Term Memory (LSTM), yaitu salah satu jenis jaringan saraf tiruan yang dirancang khusus untuk memahami pola dalam data deret waktu (time series). LSTM sangat cocok digunakan dalam kasus seperti ini karena mampu mengingat pola data dari waktu ke waktu, sehingga dapat membantu memprediksi pergerakan harga di masa depan dengan lebih akurat.

### Pemilihan Model: Long Short-Term Memory (LSTM)

LSTM dipilih karena kemampuannya dalam mengingat informasi jangka panjang serta menangani masalah vanishing gradient yang umum pada RNN biasa. Model ini dapat mengenali pola dalam data sekuensial, seperti harga pasar, yang sering memiliki korelasi antar waktu.

### Arsitektur Model

Model dibangun menggunakan Keras dengan TensorFlow backend. Berikut adalah arsitektur model awal yang digunakan:

```python
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
```
Penjelasan tahapan dan parameter:

- `LSTM Layer (64 units)` adalah jumlah neuron di layer LSTM. Lapisan ini menangkap hubungan temporal dalam data.

- `return_sequences=True` dapat mengembalikan seluruh output sequence ke layer berikutnya. Diperlukan karena ada dua layer LSTM bertingkat.

- `Dropout (0.2)` yaitu teknik regularisasi untuk mengurangi overfitting dengan mengabaikan 20% neuron secara acak selama pelatihan.

- `Dense(1)` adalah layer keluaran dengan satu neuron untuk memprediksi harga penutupan berikutnya.

Kompilasi dan Pelatihan Model:

```python
model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stop])
```
- Optimizer: `adam` Optimizer adaptif yang umum digunakan untuk deep learning.

- Loss: `mean_squared_error` cocok untuk regresi karena menghitung kesalahan rata-rata kuadrat antara prediksi dan target.

- Epochs: 50 Jumlah maksimum pelatihan iterasi.

- Batch size: 64 Jumlah data dalam satu iterasi training.

- EarlyStopping: Menghentikan pelatihan jika val_loss tidak membaik dalam 5 epoch berturut-turut.

### Arsitektur Model

Model dievaluasi menggunakan metrik:

- Mean Absolute Error (MAE)

- Mean Squared Error (MSE)

- Root Mean Squared Error (RMSE)

- R² Score

Hasil evaluasi menunjukkan performa prediksi cukup baik, berikut adalah detailnya:

- Neuron LSTM dari 64 menjadi 128
- Dropout Rate dari 0.2 menjadi 0.3
- Batch Size dari 64 menjadi 32
- Optimizer dari Adam diubah menjadi RMSprop

Model baru:

```python
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```

Terdapat peningkatan akurasi dan penurunan error pada data uji, dengan nilai RMSE dan MAE yang lebih rendah dibanding model awal.

### Kelebihan dan Kekurangan Algoritma LSTM

#### kelebihan

- Mampu mempelajari pola jangka panjang dalam data
- Cocok untuk data time series
- Menangani vanishing gradient lebih baik dari RNN
- Akurat untuk data yang memiliki pola musiman/trend

#### kekurangan

- Butuh waktu pelatihan lebih lama
- Memerlukan banyak data agar model efektif
- Kompleksitas model lebih tinggi daripada regresi biasa
- Tuning hyperparameter cukup menantang

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
#### MAE = 3,325,259.55
Rata-rata kesalahan prediksi sekitar $88, relatif kecil mengingat harga Bitcoin bisa berada di kisaran puluhan ribu USD. Ini menunjukkan akurasi yang baik.

#### RMSE = 1,695.01
Ini berarti rata-rata kesalahan prediksi model adalah sekitar $1,695. Nilai ini relatif kecil jika dibandingkan dengan harga Bitcoin yang bisa berkisar antara $20.000 hingga $60.000, sehingga menunjukkan bahwa model memiliki performa yang cukup baik dalam menghasilkan prediksi yang mendekati nilai aktual.

#### R² Score =  0.9874
Nilai ini menunjukkan bahwa sekitar 98.74% variasi harga Bitcoin dalam data historis dapat dijelaskan oleh model LSTM. Ini adalah indikasi kuat bahwa model mampu mengenali dan mempelajari pola dari data time series dengan sangat baik, dan performanya sudah sangat memadai untuk keperluan prediksi jangka pendek.









