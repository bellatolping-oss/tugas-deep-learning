# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

Tugas Workshop Deep Learning — Teknik Telekomunikasi ITERA.

## Kelompok

| Peran | Nama | NIM |
|---|---|---|
| Ketua | Isabella Silalahi | 124400068 |
| Anggota | Dwi Yanti Maharani | 124400084 |
| Anggota | Lusitania Fachria Salsabila | 123400032 |

- **Kelas:** RA
- **Lokasi kelompok:** 2 — Universita Bocconi (kampus, kotak 4460) (NIM terkecil 123400032 → digit terakhir 2)
- **Laporan:** `laporan-typst/main.pdf`

## Cara Menjalankan

```bash
uv sync
uv run jupyter lab
```

Buka `tugas_ablasi.ipynb`, set `LOKASI_KELOMPOK = 2` dan `CEPAT = False`, lalu jalankan semua sel dari atas ke bawah. Seluruh run otomatis tercatat di `hasil/ablasi.csv`. Run yang sudah ada dilewati, aman dijalankan ulang.

**Perangkat:** Laptop CPU AMD Ryzen 7 7730U (8 core / 16 thread), PyTorch 2.14.1+cpu, tanpa GPU.  
**Total waktu training:** 4982 detik ≈ **83 menit** (348 run, 3 seed per konfigurasi).

## Ringkasan Hasil

Seluruh model pembelajaran mendalam berhasil melampaui ketiga baseline naif secara signifikan. Pada konfigurasi acuan di Duomo (test MAE): ANN 309.9, RNN 229.6, LSTM 273.6, GRU 237.2. Setelah ablasi dan optimasi hyperparameter (konfigurasi terbaik), test MAE berhasil diturunkan menjadi: **ANN 257.4, RNN 226.3, LSTM 246.7, GRU 230.2** — model rekuren (RNN dan GRU) tetap unggul dibandingkan ANN. Fitur waktu trigonometrik (jam dan hari) terbukti sebagai faktor paling berpengaruh; GRU dengan hidden 128 dan jendela W=48 jam serta batch size 16 memberikan generalisasi terbaik, termasuk pada area kampus Universita Bocconi (Bagian D).

## Struktur

```
tugas_ablasi.ipynb      notebook eksperimen (di-commit beserta output)
data/                   data trafik per jam Milan (Duomo + 10 lokasi tugas)
hasil/                  CSV seluruh 348 run, tabel, dan 9 grafik ablasi
laporan-typst/          sumber laporan IEEE Typst + PDF
laporan-latex/          template alternatif LaTeX
```

## Data

Telecom Italia Big Data Challenge (Barlacchi dkk., 2015, *Scientific Data*, lisensi ODbL), versi per jam dari Zhang dkk. (INFOCOM 2021). Disiapkan oleh pengajar di repo [mctosima/dl-workshop](https://github.com/mctosima/dl-workshop).
