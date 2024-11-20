## Domain Proyek
Pada masa sekarang ini depresi menjadi jenis gangguan kejiwaan yang paling sering dialami oleh masyarakat karena tingkat stress yang sangat tinggi akibat tuntutan hidup yang semakin bertambah. Depresi merupakan gangguan mental yang serius yang ditandai dengan perasaan sedih dan cemas. Gangguan ini biasanya akan menghilang dalam beberapa hari tetapi dapat juga berkelanjutan yang dapat mempengaruhi aktivitas sehari-hari. WHO memprediksikan bahwa pada tahun 2020 depresi akan menjadi salah satu penyakit mental yang banyak dialami dan depresi berat akan menjadi penyebab kedua terbesar kematian setelah serangan jantung.[[1]](https://ejournal.gunadarma.ac.id/index.php/infokom/article/view/2418).
Untuk itu dalam proyek ini saya mengangkat judul **Klasifikasi Depresi di Kehidupan Masyarakat**. Dengan adanya model machine learning ini diharapkan dapat memudahkan pekerjaan psikolog dalam mengindetifikasi kesehatan mental.
## Business Understanding
Berdasarkan latar belakang diatas, berikut ini rumusan masalah yang dapat diselesaikan pada proyek ini:
### Problem Statements
* Bagaimana membuat model machine learning untuk memprediksi depresi atau tidak depresi?
* Bagaimana membuat algoritma yang mampu menghasilkan akurasi paling baik?

### Goals
* Membuat model machine learning untuk memprediksi depresi pada mahasiswa.
* Membuat model machine learning dengan nilai akurasi yang minimal mencapai 85%.

##### Solution Statements:
Untuk mencapai tujuan dari proyek ini, akan dibuat beberapa model yang berbeda untuk dibandingkan, diantaranya adalah menggunakan:
* **Extra Trees.**
Klasifikasi dengan metode extra trees atau yang disebut juga sebagai ―Extremly randomized Trees‖ merupakan varian pengembangan dari decision tree acak pada berbagai sub bagian dataset dan menghitung rata-ratanya untuk meningkatkan akurasi prediksi dan pengendalian over vitting. Berbeda dengan Random Forest dimana pada setiap tahapannya, sample dan keputusan diambil secara acak dan bukan diambil dari yang terbaik.[[2](https://ojs.amikom.ac.id/index.php/semnasteknomedia/article/download/1728/1456)]
* **Random Forest.**
Algoritma random forest adalah salah satu algoritma supervised learning yang dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Ia termasuk ke dalam kelompok model ensemble (group). Algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak.[[3](https://www.dicoding.com/academies/319/tutorials/18585?from=18580)]
* **Decision tree.** 
Decision tree merupakan teknik model prediksi yang dapat digunakan untuk
klasifikasi dan prediksi tugas. Decision tree menggunakan teknik “membagi dan
menaklukkan” untuk membagi ruang pencarian masalah menjadi himpunan masalah. Proses pada decision tree adalah mengubah bentuk data tabel menjadi sebuah model tree. Model tree akan menghasilkan rule dan disederhanakan.  [[4](https://jurnal.stkipalmaksum.ac.id/index.php/Sintaksis/article/download/47/57/)]

* **Adaptive boosting**
AdaBoost (Adaptive Boosting) merupakan algoritma machine learning yang dirumuskan oleh Yoav Freund dan Robert Schapire (Afza, Farid, & Rahman, 2011, p. 105) (Harrington, 2012, p. 132). Algoritama AdaBoost merupakan algoritma yang membangun pengklasifikasi kuat dengan mengombinasikan sejumlah pengklasifikasi sederhana (lemah).

### Data Understanding
Dataset yang digunakan penulis bersumber dari platform penyedia dataset yaitu Kaggle. Dataset yang digunakan berfokus tentang faktor apa saja yang menjadi penyebab depresi di kehidupan masyarakat yang tinggal di wilayah pedesaan. Dataset ini memiliki 23 kolom atau dimensi dan total 1432 baris atau objek. Berikut dataset penulis gunakan :

**Jenis**|**Keterangan**|
:-----:|:-----:
Sumber| [Kaggle dataset : depression](https://www.kaggle.com/datasets/diegobabativa/depression)|
Lisensi | GPL 2
Jenis dan Ukuran Berkas | CSV(49 kB)

Penjelasan tentang variabel yang terdapat di dataset depression ini dapat dilihat sebagai berikut :
* Survey id : untuk mengindentifikasi kandidat survey
* Ville id : untuk mengidentifikasi wilayah pedesaan/kota kandidat
* sex : Jenis kelamin kandidat
* age : Usia kandidat
* Married :  Status pernikahan kandidat
* Numberchildren  : Jumlah anak kandidat
* educationlevel : Tingkat pendidikan yang di tempuh kandidat
* totalmembers (in the family) : Jumlah anggota dalam keluarga kandidat
* gainedasset : Aset yang diperoleh oleh kandidat
* durableasset : Aset kandidat yang tahan lama
* saveasset : Aset biaya kandidat yang disimpan
* livingexpenses : pengeluaran hidup kandidat 
* otherexpenses : Biaya lainnya yang diperoleh kandidat
* incomingsalary : Gaji kandidat yang diperoleh 
* incomingownfarm : Gaji kandidat yang menernak hewan
* incomingbusiness : Gaji kandidat yang berbisnis
* incomingnobusiness : Gaji kandidat yang didapat selain berbisnis
* incomingagricultural : Gaji kandidat yang bertani
* farmexpenses : Biaya pertanian yang dikeluarkan oleh kandidat yang bekerja sebagai petani
* laborprimary : Tenaga kerja  yang dimiliki kandidat
* lastinginvestment : Kandidat yang berinvestasi seumur hidup
* nolastinginvestmen : Kandidat yang tidak berinvestasi seumur hidup
* depressed : Kandidat yang mengalami depresi

Berikut beberapa tahapan visualisasi data :
1. Melakukan visualisasi data dalam bentuk sns.countplot.Penggunaan sns.countplot sendiri berfungsi untuk menghitung jumlah data yang sama.u 
![Gambar 1](https://i.postimg.cc/tR1gWYsy/Visual1.png) 
Pada beberapa variable plot diatas bisa dilihat bahwa kandidat yang tidak mengalami depresi cukup tinggi dibandingkan dengan yang terkena depresi

2. Melakukan visualisasi distribusi numerik, yg dapat dilihat lebih rinci sebagai berikut: 
![Gambar 2](https://i.postimg.cc/4xwKwjsT/Visual2.png)

3. Kemudian melakukan visualisasi distribusi categorial, dimana ini digunakan untuk menghitung jumlah sample kandidat yang depresi (1) dan kandidat yang tidak mengalami depresi (0). pada project ini terdapat 1191 jumlah data sampel kandidat yang tidak mengalami depresi (0) dan 238 jumlah data sampel kandidat yang depresi (0)
5. Selanjutnya visualisasi dilakukan untuk mengetahui korelasi antar fitur yang terdapat pada dataset, untuk selengkapnya sebagai berikut
![Gambar 3](https://i.postimg.cc/mgq4bSSH/Visual3.png)
Setelah dibuat grafik ini, kita telah mendapat banyak data yang memudahkan proses analisis data.

### Data Preparation 
Sebelum menjalankan tahap persiapan data, penulis perlu melakukan beberapa langkah.
1. ```df.info()``` digunakan untuk mengecek tipe kolom pada dataset
2. ```df.isna().sum()``` digunakan untuk mengecek apakah ada kolom yg kosong, pada dataset ini nilai kosong tidak ditemukan
3. ```df.describe()``` digunakan utk mendapatkan info mengenai dataset terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain.
4. Hapus fitur yang tidak digunakan, langkah ini dilakukan untuk meminimalkan kolom seperti berikut yang terkesan berlebihan dan tidak secara signifikan mempengaruhi tujuan proyek ini. Untuk drop filenya seperti berikut 'Survey_id', 'Ville_id', 'gained_asset', 'durable_asset', 'save_asset', 'other_expenses', 'incoming_agricultural', 'farm_expenses', 'labor_primary', 'lasting_investment', 'no_lasting_investmen'.

Untuk persiapan data, penulis menggunakan beberapa teknik yang diperlukan dalam tahap persiapan data. Sebagai berikut:
- Melakukan perhitungan jumlah baris terhadap kolom target
- Train-Test-Split : Melakukan pembagian dataset menjadi dengan 80% untuk data latih dan 20% untuk data uji Setelah melakukan pra-pemrosesan ke dataset, Data latih adalah data yang hanya digunakan untuk melatih model, sedangkan data uji adalah data yang hanya digunakan sebagai ujicoba model. Pembagian dataset ini menggunakan modul train_test_split dari scikit-learn.
- Standarisasi : Langkah terakhir adalah standarisasi data. Hal ini dimaksudkan agar semua fitur memiliki skala data yang sama (berkisar dari 0 sampai 1). Untuk merangkai data ini, gunakan fungsi StandardScaler. 

### Modeling
Setelah melakukan preprocessing dataset, langkah selanjutnya adalah memodelkan data. Empat algoritma digunakan pada tahap ini: Random Forest, Extra Tree, Decision Tree dan Adaptive boosting. Pertama, keempat model ini dilatih pada data pelatihan. Kedua model tersebut kemudian diuji dengan data uji. Terakhir, ukur keakuratan keempat model tersebut. 
Berikut nilai parameter dan parameter dari ke eempat model
- Random Forest : Parameter yang cocok untuk model Random Forest adalah ```n_estimators``` = 100, ```max_depth```=None, ```min_samples_split```=2, ```random_state```=0, ```n_jobs```=-1 dan hasil akurasi prediksi yang dihasilkan ialah **0.8671328671328671**
- Decision Tree : Parameter yang cocok untuk model Decision Tree adalah ```max_depth```=None, ```min_samples_split```=2,```random_state```=0 dan hasil akurasi prediksi yang dihasilkan ialah **0.7727272727272727**
- Extra Tree : Parameter yang cocok untuk model Extra Tree adalah ```n_estimators``` = 100, ```max_depth```=None, ```min_samples_split```=2, ```random_state```=0, ```n_jobs```=-1 dan hasil akurasi prediksi yang dihasilkan ialah **0.8391608391608392**
- Adaptive boosting : Parameter yang cocok untuk model Adaptive boosting adalah ```n_estimators``` = 100, ```random_state```=0 dan hasil akurasi prediksi yang dihasilkan ialah **0.8916083916083916**

Berikut perbandingan hasil keempat model tersebut.

|              |              |              | 0             |            |              | 1             |            |
|--------------|--------------|--------------|---------------|------------|--------------|---------------|------------|
|              | **accuracy** | **f1-score** | **precision** | **recall** | **f1-score** | **precision** | **recall** |
| RandomForest |   0.867133   |   0.928571   |    0.888489   |  0.972441  |   0.050000   |    0.125000   |   0.03125  |
| ExtraTrees   |   0.839161   |   0.911877   |    0.888060   |  0.937008  |   0.080000   |    0.111111   |   0.06250  |
| DecisionTree |   0.772727   |   0.867076   |    0.902128   |  0.834646  |   0.216867   |    0.176471   |   0.28125  |
| AdaBoost     |   0.891608   |   0.942486   |    0.891228   |  1.000000  |   0.060606   |    1.000000   |   0.03125  |

Pada model dengan algoritma Adaptive boosting memiliki nilai akurasi, f1-score, recall dan precision sedikit lebih tinggi dibanding dengan algoritma Random Forest, Extra Tree, dan Decision Tree. 

### Evaluation
Proyek ini menggunakan empat metrik. Empat metrik tersebut adalah:
* Accuracy: Ratio dari True Positives dan True Negative terhadap seluruh positif dan negative di seluruh observasi. Rumus Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
* Precision: Kemampuan model untuk memprediksi nilai positif terhadap seluruh jumlah positif yang diprediksi oleh model. Rumus Precision Score = TP / (FP + TP)
* Recall: Kemampuan model untuk memprediksi nilai positif terhadap seluruh jumlah positif yang sesungguhnya. Rumus Recall Score = TP / (FN + TP)
* F1: Metrik yang menimbang kemampuan model untuk memberikan Precision dan Recall. Rumus F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)

Keterangan 

* True Positive (TP): Jumlah prediksi positif yang benar terhadap jumlah positif yang sebenarnya.
* False Positive (FP): Jumlah prediksi positif yang salah.
* True Negative (TN): Jumlah prediksi negatif yang benar terhadap jumlah negatif yang sebenarnya.
* False Negative (FN): Jumlah prediksi negatif yang salah.

Berikut adalah tabel evaluasi dari keempat matriks dari model Adaptive boosting.

|              | precision |  recall  | f1-score |   support  |
|--------------|:---------:|:--------:|:--------:|:----------:|
| 0            |  0.891228 | 1.000000 | 0.942486 | 254.000000 |
| 1            |  1.000000 | 0.031250 | 0.060606 |  32.000000 |
| accuracy     |  0.891608 | 0.891608 | 0.891608 |  0.891608  |
| macro avg    |  0.945614 | 0.515625 | 0.501546 | 286.000000 |
| weighted avg |  0.903398 | 0.891608 | 0.843814 | 286.000000 |

### Link Streamlit
[Dashboard](https://predictiveanalyticsdepression-eruwhwpnappahmgp7fajhwf.streamlit.app/)
