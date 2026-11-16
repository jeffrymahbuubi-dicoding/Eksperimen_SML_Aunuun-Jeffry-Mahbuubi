# Eksperimen_SML_Aunuun-Jeffry-Mahbuubi

Eksperimen dan preprocessing otomatis untuk dataset **AI4I 2020 Predictive Maintenance**
(UCI Machine Learning Repository, id 601, lisensi CC BY 4.0). Targetnya adalah `Machine failure`.

## Struktur

```
ai4i2020_raw/                         data mentah (ai4i2020.csv)
preprocessing/
  Eksperimen_Aunuun-Jeffry-Mahbuubi.ipynb           notebook eksperimen: memuat data, EDA, preprocessing
  automate_Aunuun-Jeffry-Mahbuubi.py                preprocessing otomatis dalam satu skrip
  ai4i2020_preprocessing/             hasil: train.csv, test.csv, preprocessor.joblib, columns.json
.github/workflows/preprocessing.yml   menjalankan skrip otomatis dan menyimpan hasilnya
requirements.txt
```

## Preprocessing

Kolom identitas (`UDI`, `Product ID`) dan lima kolom mode kegagalan (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`)
dibuang. Kolom mode kegagalan menentukan nilai target, sehingga memakainya sebagai fitur menjadi kebocoran
data. Data dibagi 80/20 secara stratifikasi, kolom `Type` di-encode one-hot, dan lima kolom numerik
distandarisasi dengan statistik dari data latih saja.

## Menjalankan

```
pip install -r requirements.txt
python preprocessing/automate_Aunuun-Jeffry-Mahbuubi.py --raw ai4i2020_raw/ai4i2020.csv --out preprocessing/ai4i2020_preprocessing
```

Workflow GitHub Actions menjalankan perintah yang sama setiap kali data mentah atau skrip berubah, atau
lewat tombol "Run workflow", lalu meng-commit hasilnya.
