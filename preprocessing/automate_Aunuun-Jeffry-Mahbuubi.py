#!/usr/bin/env python3
"""Preprocessing otomatis dataset AI4I 2020 Predictive Maintenance.

Langkah-langkah sama dengan notebook Eksperimen_Aunuun-Jeffry-Mahbuubi.ipynb:
  1. Nilai kosong dan duplikat: diperiksa dan dilaporkan, tidak ada baris yang dihapus.
  2. Membuang kolom identitas (UDI, Product ID).
  3. Membuang kolom mode kegagalan (TWF, HDF, PWF, OSF, RNF): kolom itu menentukan nilai
     target sehingga menjadi kebocoran data.
  4. Pembagian data 80% latih dan 20% uji, stratifikasi pada target, random_state=42.
  5. Encoding kolom Type dengan one-hot (Type_H, Type_L, Type_M).
  6. Standarisasi lima kolom numerik dengan StandardScaler.
  Langkah 5 dan 6 dipelajari hanya dari data latih, lalu dipakai untuk data uji.

Pemakaian:
    python automate_Aunuun-Jeffry-Mahbuubi.py --raw ../ai4i2020_raw/ai4i2020.csv --out ai4i2020_preprocessing

Keluaran di folder --out: train.csv, test.csv, preprocessor.joblib, columns.json.
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Machine failure"
KOLOM_IDENTITAS = ["UDI", "Product ID"]
KOLOM_MODE_KEGAGALAN = ["TWF", "HDF", "PWF", "OSF", "RNF"]
KOLOM_NUMERIK = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                 "Torque [Nm]", "Tool wear [min]"]
KOLOM_KATEGORI = "Type"
KATEGORI_TYPE = ["H", "L", "M"]
NAMA_NUMERIK = ["Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
                "Torque_Nm", "Tool_wear_min"]
KOLOM_HASIL = NAMA_NUMERIK + ["Type_H", "Type_L", "Type_M", "Machine_failure"]
UKURAN_UJI = 0.2
RANDOM_STATE = 42


def buat_preprocessor():
    """Standarisasi kolom numerik dan one-hot kolom Type dengan urutan kategori tetap."""
    return ColumnTransformer(
        [
            ("numerik", StandardScaler(), KOLOM_NUMERIK),
            ("type", OneHotEncoder(categories=[KATEGORI_TYPE], sparse_output=False), [KOLOM_KATEGORI]),
        ]
    )


def _ke_frame(preprocessor, fitur, target):
    matriks = preprocessor.transform(fitur)
    frame = pd.DataFrame(matriks, columns=KOLOM_HASIL[:-1], index=fitur.index)
    frame["Machine_failure"] = target.astype(int).to_numpy()
    return frame


def preprocess(raw_csv, out_dir):
    """Jalankan seluruh preprocessing dan tulis hasilnya ke out_dir.

    Mengembalikan dict berisi jumlah baris, jumlah kolom dan porsi kelas 1 tiap bagian.
    """
    # berkas mentah diawali BOM UTF-8, maka dibaca dengan utf-8-sig
    data = pd.read_csv(raw_csv, encoding="utf-8-sig")
    # langkah 1: laporan nilai kosong dan duplikat; data tidak diubah
    print("Nilai kosong: %d | Baris duplikat: %d"
          % (int(data.isna().sum().sum()), int(data.duplicated().sum())))
    data = data.drop(columns=KOLOM_IDENTITAS + KOLOM_MODE_KEGAGALAN)

    fitur = data.drop(columns=[TARGET])
    target = data[TARGET]
    x_latih, x_uji, y_latih, y_uji = train_test_split(
        fitur, target, test_size=UKURAN_UJI, random_state=RANDOM_STATE, stratify=target
    )

    preprocessor = buat_preprocessor()
    preprocessor.fit(x_latih)  # hanya data latih
    latih = _ke_frame(preprocessor, x_latih, y_latih)
    uji = _ke_frame(preprocessor, x_uji, y_uji)

    os.makedirs(out_dir, exist_ok=True)
    latih.to_csv(os.path.join(out_dir, "train.csv"), index=False, lineterminator="\n")
    uji.to_csv(os.path.join(out_dir, "test.csv"), index=False, lineterminator="\n")
    joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.joblib"))
    with open(os.path.join(out_dir, "columns.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"fitur": KOLOM_HASIL[:-1], "target": "Machine_failure"}, f, indent=2)
        f.write("\n")

    ringkasan = {
        "train_rows": len(latih),
        "test_rows": len(uji),
        "columns": len(KOLOM_HASIL),
        "train_failure_share": float(latih["Machine_failure"].mean()),
        "test_failure_share": float(uji["Machine_failure"].mean()),
    }
    print("Data latih: %d baris | Data uji: %d baris | Kolom: %d"
          % (ringkasan["train_rows"], ringkasan["test_rows"], ringkasan["columns"]))
    print("Porsi kelas 1 | latih: %.4f | uji: %.4f"
          % (ringkasan["train_failure_share"], ringkasan["test_failure_share"]))
    return ringkasan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Preprocessing dataset AI4I 2020 Predictive Maintenance.")
    ap.add_argument("--raw", required=True, help="berkas CSV mentah ai4i2020.csv")
    ap.add_argument("--out", required=True, help="folder keluaran")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.raw):
        print("Berkas mentah tidak ditemukan: %s" % args.raw, file=sys.stderr)
        return 1
    preprocess(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
