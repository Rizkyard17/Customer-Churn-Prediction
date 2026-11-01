# 📉 Predictive Analytics: Telco Customer Churn

Proyek *machine learning* untuk memprediksi pelanggan perusahaan telekomunikasi yang berpotensi **berhenti berlangganan (churn)**. Proyek ini membandingkan empat algoritma klasifikasi dan menerjemahkan hasilnya menjadi rekomendasi bisnis untuk strategi retensi pelanggan.

---

## 📌 Latar Belakang

Mempertahankan pelanggan yang sudah ada umumnya jauh lebih murah daripada mengakuisisi pelanggan baru. Dengan mengetahui pelanggan mana yang berisiko churn, tim bisnis dapat memberikan intervensi lebih awal, misalnya promo retensi atau penawaran kontrak jangka panjang.

## 🎯 Tujuan

1. Membangun model klasifikasi biner untuk memprediksi `Churn` (Yes/No).
2. Membandingkan performa Logistic Regression, Random Forest, XGBoost, dan LightGBM.
3. Mengidentifikasi faktor utama yang memengaruhi churn.
4. Menghasilkan *actionable insights* untuk strategi retensi.

## 📂 Dataset

- **Nama:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Sumber:** [Kaggle](https://www.kaggle.com/code/farazrahman/telco-customer-churn-logisticregression?select=WA_Fn-UseC_-Telco-Customer-Churn.csv)
- **Target:** `Churn` (Yes/No)
- **Contoh fitur:** `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, `InternetService`, `OnlineSecurity`, `TechSupport`, `Partner`, dan lainnya.

## 🔄 Alur Kerja

```
Import Data → Preprocessing → EDA → Encoding → Split & Scaling → SMOTE → Modeling → Evaluasi → Insight
```

### 1. Data Preprocessing
- Pengecekan tipe data, statistik deskriptif, *missing value*, dan data duplikat.
- Kolom `TotalCharges` dikonversi ke numerik. Nilai kosong diisi `0` karena berasal dari pelanggan baru dengan `tenure = 0`.

### 2. Exploratory Data Analysis (EDA)
- Visualisasi distribusi setiap variabel kategorikal menggunakan `countplot` (Seaborn).

### 3. Encoding
| Kolom | Metode |
|---|---|
| `Churn` | Label Encoding |
| `PaymentMethod` | Binary Encoding (`category_encoders`) |
| `Contract` | Ordinal mapping (Month-to-month = 0, One year = 1, Two year = 2) |
| Kolom kategorikal lainnya | One-Hot Encoding (`drop_first=True`) |

### 4. Splitting, Scaling, dan Penanganan Imbalance
- *Train-test split* **80:20** dengan `random_state=42`.
- `StandardScaler` pada `tenure`, `MonthlyCharges`, dan `TotalCharges` (*fit* hanya pada data latih untuk mencegah *data leakage*).
- **SMOTE** untuk menyeimbangkan kelas pada data latih.

### 5. Modeling
Empat model dilatih pada data hasil SMOTE:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

### 6. Evaluasi
- *Confusion matrix*, *classification report*, dan **ROC-AUC** pada data uji.
- Kurva ROC gabungan untuk membandingkan seluruh model.
- *Feature importance* dari Random Forest.

## 📊 Hasil

### Perbandingan Model (ROC-AUC)

| Peringkat | Model | AUC |
|:-:|---|:-:|
| 🥇 | **Logistic Regression** | **0.85** |
| 2 | Random Forest | 0.84 |
| 3 | LightGBM | 0.84 |
| 4 | XGBoost | 0.83 |

**Logistic Regression** menjadi model terbaik dan direkomendasikan karena:
- AUC tertinggi.
- Komputasi lebih ringan dibanding model *ensemble*.
- Koefisien mudah diinterpretasikan untuk kebutuhan bisnis.

### Fitur Paling Berpengaruh (Random Forest)

| Fitur | Skor Importance |
|---|:-:|
| `TotalCharges` | ~0.104 |
| `tenure` | ~0.098 |
| `MonthlyCharges` | ~0.086 |
| `Contract` | ~0.078 |
| `OnlineSecurity_Yes` | ~0.033 |

Fitur lain seperti `InternetService_Fiber optic`, `PaymentMethod_1`, `TechSupport_Yes`, `Partner_Yes`, dan `OnlineBackup_Yes` memberi kontribusi lebih kecil (< 0.03).

## 💡 Wawasan Bisnis

1. **Intervensi finansial:** beri perhatian atau promo retensi untuk pelanggan baru dengan `MonthlyCharges` tinggi dan `tenure` rendah.
2. **Edukasi kontrak:** dorong pelanggan *month-to-month* beralih ke kontrak 1 atau 2 tahun melalui diskon khusus.
3. **Optimasi layanan:** tingkatkan penjualan atau *bundling* layanan tambahan seperti `OnlineSecurity` dan `TechSupport` untuk memperkuat keterikatan pelanggan.

## 🛠️ Tech Stack

- **Bahasa:** Python
- **Data & visualisasi:** pandas, NumPy, Matplotlib, Seaborn
- **Preprocessing:** scikit-learn, category_encoders, imbalanced-learn (SMOTE)
- **Modeling:** scikit-learn, XGBoost, LightGBM
- **Environment:** Google Colab

## 🚀 Cara Menjalankan

1. Clone repositori ini.
   ```bash
   git clone https://github.com/<username>/<nama-repo>.git
   cd <nama-repo>
   ```
2. Install dependensi.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn category_encoders xgboost lightgbm
   ```
3. Unduh dataset dari Kaggle dan letakkan di folder proyek.
4. Buka notebook `_Portfolio__Predictive_Analytics_Churn.ipynb` di Jupyter atau Google Colab.
5. Sesuaikan path dataset, lalu jalankan semua sel.

> **Catatan:** notebook asli memuat dataset dari Google Drive (`/content/drive/MyDrive/Dataset/...`). Jika dijalankan secara lokal, ganti dengan `pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')` dan hapus bagian `drive.mount`.

## 🔭 Pengembangan Selanjutnya

- *Hyperparameter tuning* dengan `GridSearchCV`.
- Evaluasi dengan *cross-validation* dan metrik tambahan (Precision, Recall, F1) untuk kelas churn.
- Analisis interpretabilitas dengan SHAP.
- *Deployment* model sebagai API atau dashboard sederhana.

## 📁 Struktur Proyek

```
├── _Portfolio__Predictive_Analytics_Churn.ipynb
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   # unduh dari Kaggle
```

## 👤 Author

**M. Rizky Ardiansyah Putra**
- LinkedIn: [link]
- GitHub: [link]
- Email: [email]
