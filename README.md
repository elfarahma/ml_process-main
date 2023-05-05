# Description of Project
## Background
Project machine learning ini bertujuan untuk memprediksi nilai ecological footprint suatu negara dengan mempertimbangkan faktor-faktor seperti Indeks Pembangunan Manusia (HDI) dan lokasi geografis di benua mana negara tersebut berada. Metode yang digunakan adalah regresi linier dengan input berupa HDI dan data kategorikal berupa benua. Output yang dihasilkan adalah prediksi nilai ecological footprint suatu negara. Proyek ini dapat memberikan wawasan tentang faktor-faktor yang berkontribusi pada tingkat kerusakan lingkungan dan dapat membantu dalam merencanakan kebijakan lingkungan yang lebih berkelanjutan di masa depan.

**Objective**

Dengan demikian objektif dari final project ini juga adalah membangun model berbasis machine learning yang dapat memprediksi nilai ecological footprint penduduk suatu negara,  dengan menggunakan input demografi (misalnya seperti HDI dan harapan hidup) dan fitur kewilayahan suatu negara (benua).  

**Business Metrics**

Metriks bisnis yang dapat diambil adalah upaya dan waktu untuk menganalisa sumber-sumber kontribusi peningkatan jejak korban dalam rangka menekan laju kerusakan lingkungan dapat dipermudah dan dipersingkat.

## Project Architecture

Proyek ini bertujuan untuk mengembangkan model machine learning untuk memprediksi suatu variabel target. Prosesnya terdiri dari langkah-langkah berikut:

1. **Persiapan Data**

   Pada langkah ini, akan dilakukan kegiatan-kegiatan berikut:

   - Pengumpulan data
   - Pendefinisian data, yaitu menentukan lingkup dan batasan nilai data yang meliputi rentang nilai pada kolom numerikal (HDI, EFConsPerCap), tipe data (str, int, float, dll), serta batasan kelas pada kolom kategorikal (continent).
   - Validasi data, yakni memastikan bahwa setiap entry pada dataset sudah sesuai dengan batasan yang ditentukan dalam pendefinisian data.
   - Data defense, mekanisme warning apabila ada entry data dari API yang tidak sesuai dengan pendefinisian data.
   - Data splitting, membagi dataset untuk tujuan training, test, dan validasi. Test size yang digunakan adalah 20%.

2. **Exploratory Data Analysis (EDA)**

   Tahapan EDA dilakukan untuk menganalisa apakah dataset yang ada memiliki kecenderungan yang akan menjadi penghambat atau mengurangi performa model, untuk kemudian dianalisa modifikasi apa yang tepat untuk data tersebut. Hal-hal yang dianalisis meliputi: adanya null-data, skewness, dan imbalance pada data.

3. **Preprocessing**

   Proses yang dilakukan dalam tahap ini adalah penanganan terhadap null-data dengan melakukan imputasi berdasarkan hasil EDA, dan feature engineering. Untuk feature engineering, proses yang dilakukan adalah:

   - Transformasi data kategorikal (continent) dengan menggunakan one hot encoding.
   - Standardisasi data.
   - Penanganan imbalance data dengan membuat 3 dataset dengan tehnik yang berbeda yaitu undersampling, oversampling, dan SMOTE.

4. **Modeling**

   Dalam tahapan ini, ada tiga model regresi yang akana dievaluasi untuk kemudian dipilih sebagai model produksi, yaitu regresi linear, Random Forest Regressor, dan Tree Decision Regressor. Proses yang dilakukan dalam tahapan ini adalah:

   - Training dan evaluasi model, dimana di akhir proses ini ditentukan model yang terbaik yang akan digunakan sebagai model produksi sesuai dengan metriks MSE (nilai terendah), R-Square (nilai tertinggi), training time (tersingkat).
   - Optimalisasi dengan melakukan hyper-parameter tuning, untuk kemudian dilakukan kembali tahap training dan evaluasi model, serta pemilihan model produksi.
   - Dokumentasi hasil training dan evaluasi model ke dalam training log.

5. **Deployment**

   Deployment merupakan proses kloning infrastruktur machine learning dari environment pada perangkat computer host ke environment baru agar dapat diakses oleh pengguna lain dan dari mana saja dengan menggunakan API. Deployment pada final project ini akan memanfaatkan docker sebagai kontainer pada server AWS.

## Output yang Diharapkan

- Hasil prediksi ML dari model regresi dengan performa terbaik.
- Training log yang merekam 
   - Metriks performa model: Mean Squared Error (MSE), R-Square, dan training time
   - Penanganan imbalance data: undersampling, oversampling, SMOTE 

# Documentation


## Format Data untuk Prediksi melalui API

1. Format data: API ini memerlukan data masukan dalam format CSV.

2. Header kolom: File CSV harus memiliki header kolom untuk setiap fitur yang digunakan untuk prediksi. Dalam hal ini, kolom yang diperlukan adalah "hdia" dan "continent".

3. Kolom label: File CSV juga harus memiliki kolom label dengan nama "EFConSPerCap". Ini adalah kolom yang akan digunakan oleh API untuk membuat prediksi.

4. Jenis data: Kolom "hdi" harus berisi data numerik (float), sedangkan kolom "continent" harus berisi data kategorikal. Kolom "EFConsPerCap" harus berisi data numerik (float).

5. Nilai yang hilang: API tidak menerima nilai yang hilang. Pastikan semua nilai ada dan valid.

6. Encoding: File CSV harus di-encode dalam format UTF-8.

## Format Prediksi dari API

- Nilai prediksi Ecological Footprint per Capita

## Workflow

![My Image](https://github.com/elfarahma/ml_process/blob/9a710b8c613ff38bddf6467da5364de29ecb18fa/ML_Process.jpg)


# Deskripsi Dataset

Dataset yang digunakan dalam proyek ini adalah dataset time series tahunan dengan rentang waktu dari tahun 2000 sampai dengan 2014 yang mencakup data profil negara-negara seluruh dunia. Total entry yang termaktub dalam dataset tersebut adalah 2156 poin data. Dataset ini merupakan data repository Github yang diolah dengan menggabungkan data dari World Bank dan data dari Global Footprint Network (Shropshire, 2019).

Profil negara dalam dataset ini terdiri dari beberapa variabel yaitu:

- **Country**: mencakup 146 negara di seluruh dunia.
- **Continent**: mencakup 6 wilayah benua yaitu Asia, Europe, Africa, South America, Oceania, dan North America.
- **HDI (Human Development Index)**: parameter kependudukan yang merangkum tingkat kesejahteraan masyarakat suatu wilayah dari aspek kesehatan, pendidikan, dan taraf hidup masyarakat, dengan skala 0 – 1.
- **Life Expectancy**: rata-rata harapan hidup penduduk suatu negara.
- **Population**: jumlah penduduk suatu negara pada tahun ke-n.
- **Ecological Footprint per Capita (EFConsPerCap)**: rata-rata besaran porsi kapasitas lingkungan per tahun yang dihabiskan per penduduk suatu negara dalam melangsungkan kehidupannya, mencakup semua kebutuhan baik primer, sekunder, ataupun tersier (Global Hectare per capita).
- **Total Ecological Footprint dalam Global Hectare (EFConsTotGHA)**: total besaran porsi kapasitas lingkungan yang dihabiskan suatu negara (Total GHA).
- **Biocapacity per Capita (BiocapPerCap)**: kapasitas lingkungan suatu negara dalam penyediaan sumberdaya dalam pemenuhan gaya hidup per penduduk (GHA per capita).
- **Total Biocapacity dalam Global Hectare (BiocapTotGHA)**: total kapasitas lingkungan suatu negara dalam penyediaan sumberdaya dalam pemenuhan gaya hidup per penduduk (GHA per capita).

Dari dataset di atas, fitur yang digunakan untuk memprediksi nilai Ecological Footprint suatu negara adalah:

- HDI ("hdi")
- Continent ("continent")

Sedangkan untuk parameter ecological footprint yang akan diprediksi adalah Ecological Footprint Per Capita (EFConsPerCap).

# Bagaimana menggunakannya
Untuk menggunakan model machine learning ini, Anda dapat mengirimkan permintaan POST ke API endpoint dengan payload input sebagai berikut:

```
{
    "hdi": 0.7,
    "continent": "Asia"
}

```
# Format Project Output

Bila berhasil, maka Anda akan mendapat hasil prediksi dengan format berikut:

```
{
    "res": 5.7
}

```
# Kesimpulan dan Referensi

## Kesimpulan
- Setelah dilakukan sesi training dan evaluasi dengan tiga model regresi. Didapat bahwa model regresi dengan performa terbaik adalah Random Forest Regressor dengan metode SMOTE untuk penanganan imbalance data. Model training  dengan hyper-parameter tuning menunjukkan performa dari metriks MSE dan R-square yang lebih baik daripada yang tidak. Namun waktu latih bertambah  cukup signifikan. 

- Outlier pada dataset ini sangat terkait dengan fitur “continent”. Pengklasifikasian wilayah ke dalam kategori benua adalah pengkerucutan dari kategori negara. Namun pengekerucutan ini dapat dikatakan terlalu sederhana dan tidak representatif. Fitur benua pada kategori Eropa menunjukkan 70% data berada pada ranah HDI yang tinggi, tetapi tidak berdistribusi normal dan condong (skewing) sebelah kanan. Ini menunjukkan adanya outlier pada sebelah kiri. Hal serupa juga ditunjukkan pada hubungan antara fitur benua dan nilai EF per kapita, terlihat bahwa semua kategori memiliki kecenderungan condong ke sebelah kiri.

- Untuk pengembangan dan peningkatan performa model, perlu melakukan pengecilan ruang lingkup model ML dalam project ini. Dimana skalanya diperkecil dari skala "world" menjadi "region" contohnya Asia Tenggara, Eropa Tengah, Afrika Utara, dll. Dengan demikian karakteristik suatu negara dapat lebih tertangkap, namun jumlah kelas tidak terlalu besar.


## Referensi

Shropshire, A. (2019). HDI vs Ecology Plotly. Github. https://github.com/as6140/hdi_v_ecology_plotly_blog
