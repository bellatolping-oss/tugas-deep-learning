// =====================================================================
// Laporan Tugas — Workshop Deep Learning (Teknik Telekomunikasi ITERA)
// Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler
// Format IEEE conference dua kolom (paket charged-ieee)
// =====================================================================
#import "@preview/charged-ieee:0.1.4": ieee

#set text(lang: "id")

#show: ieee.with(
  title: [Studi Ablasi RNN, LSTM, dan GRU dibandingkan ANN untuk Prediksi Trafik Seluler Kota Milan],
  abstract: [
    Prediksi beban trafik internet seluler per jam memegang peranan krusial bagi operator telekomunikasi dalam manajemen sumber daya radio dinamis dan efisiensi energi. Studi ini menyajikan investigasi empiris komprehensif melalui studi ablasi sistematis untuk membandingkan empat arsitektur: Artificial Neural Network (ANN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), dan Gated Recurrent Unit (GRU), menggunakan dataset Telecom Italia pada kawasan Duomo (kotak 5060) dan kampus Universita Bocconi (kotak 4460). Tujuh faktor hyperparameter dievaluasi: panjang jendela historis, ukuran hidden state, kedalaman lapisan, learning rate, komposisi fitur, batch size, dan fungsi loss, dengan 3 random seed per konfigurasi. Pada konfigurasi acuan ($W=24$), test MAE Duomo adalah: ANN $309.9 plus.minus 5.9$, RNN $229.6 plus.minus 5.1$, LSTM $273.6 plus.minus 25.4$, GRU $237.2 plus.minus 9.3$. Setelah optimasi ablasi, konfigurasi terbaik menurunkan test MAE menjadi ANN $257.4 plus.minus 8.0$, RNN $226.3 plus.minus 4.9$, LSTM $246.7 plus.minus 12.0$, GRU $230.2 plus.minus 8.5$. Fitur waktu trigonometrik terbukti sebagai faktor paling berpengaruh; penumpukan lapisan rekuren di atas satu lapis kontraproduktif pada data terbatas. Total 348 run dieksekusi dalam 83 menit pada CPU AMD Ryzen 7 7730U.
  ],
  authors: (
    (name: "Isabella Silalahi", department: [NIM 124400068], organization: [Teknik Telekomunikasi, ITERA], email: "isabella.124400068@student.itera.ac.id"),
    (name: "Dwi Yanti Maharani", department: [NIM 124400084], organization: [Teknik Telekomunikasi, ITERA], email: "dwiyanti.124400084@student.itera.ac.id"),
    (name: "Lusitania Fachria Salsabila", department: [NIM 123400032], organization: [Teknik Telekomunikasi, ITERA], email: "lusitania.123400032@student.itera.ac.id"),
  ),
  index-terms: ("prediksi trafik seluler", "studi ablasi", "deep learning", "recurrent neural network", "LSTM", "GRU", "deret waktu"),
  bibliography: bibliography("references.bib", title: [Daftar Pustaka]),
  figure-supplement: [Gambar],
)

= Pendahuluan
Pertumbuhan eksponensial konsumsi data seluler mendorong tantangan signifikan pada infrastruktur telekomunikasi. Kemampuan memprediksi trafik seluler secara akurat sangat dibutuhkan untuk manajemen bandwidth adaptif dan alokasi sumber daya optimal @zhang2021fedda. Prediksi trafik merupakan persoalan deret waktu non-linier yang dipengaruhi siklus aktivitas manusia harian dan mingguan.

ANN feedforward memperlakukan jendela input sebagai vektor statis, mengabaikan urutan sekuensial dan menyebabkan parameter membengkak seiring panjang jendela. Sebaliknya, RNN @hochreiter1997lstm, LSTM @hochreiter1997lstm, dan GRU @cho2014gru memproses urutan secara bertahap dengan hidden state internal. Meskipun model rekuren secara teoretis superior, kinerjanya sangat sensitif terhadap pemilihan hyperparameter. Studi ablasi menjadi pendekatan terkontrol untuk mengidentifikasi faktor yang benar-benar berkontribusi.

Penggunaan dataset trafik Kota Milan menjadi tolok ukur standar dalam riset ini karena mampu merepresentasikan dinamika perkotaan yang kompleks. Bagi operator seluler, kemampuan prediksi trafik yang akurat tidak hanya meningkatkan *Quality of Service*, tetapi juga memungkinkan penerapan mekanisme *cell sleeping* secara adaptif pada *Base Transceiver Station* untuk efisiensi energi.

Kontribusi utama makalah ini:
+ Studi ablasi sistematis terhadap 7 faktor hyperparameter (A1–A5 wajib, B1 dan B4 pilihan) pada 4 model dengan 3 seed, total 348 run.
+ Identifikasi konfigurasi optimal per model berbasis val MAE dan pengujian robustnessnya pada minggu Natal.
+ Uji generalisasi spasial konfigurasi terbaik pada area kampus Universita Bocconi (kotak 4460).

= Data dan Metode

== Dataset dan Karakteristik Trafik
Dataset bersumber dari Telecom Italia Big Data Challenge (2013) @barlacchi2015milan versi per jam oleh Zhang dkk. @zhang2021fedda. Data mencakup rekaman aktivitas telekomunikasi di kota Milan per grid $200 times 200$ meter, meliputi: trafik internet (`net`), panggilan suara (`call`), dan pesan teks (`sms`).

Dua lokasi grid:
1. *Duomo* (kotak 5060): Pusat kota komersial, volume trafik sangat tinggi.
2. *Universita Bocconi* (kotak 4460, kode 2): Area kampus dengan puncak perkuliahan dan penurunan tajam saat libur.

#figure(
  image("gambar/peta_lokasi_tugas.png", width: 85%),
  caption: [Peta rata-rata aktivitas internet di Milan. Lingkaran toska: Duomo. Lingkaran kuning: lokasi tugas.],
  placement: top,
) <fig:peta>

== Pembagian Temporal dan Praproses
Rentang data: 4 November – 29 Desember 2013 (1.344 jam). Pembagian kronologis ketat:
- *Train*: 11 Nov – 8 Des (672 jam)
- *Val*: 9 – 15 Des (168 jam)
- *Test*: 16 – 22 Des (168 jam)
- *Natal*: 23 – 29 Des (168 jam, anomali perilaku)

Standarisasi z-score menggunakan statistik dari data sebelum periode validasi. Fitur temporal siklis: $sin/cos$ dari jam ($h in [0,23]$) dan hari ($d in [0,6]$).

== Arsitektur Model dan Konfigurasi Acuan
Empat model diimplementasikan menggunakan PyTorch @paszke2019pytorch. Konfigurasi acuan dirancang agar parameter rekuren setara (~5.300–6.500 parameter), sebagaimana disajikan pada @tab:acuan.

#figure(
  caption: [Konfigurasi hyperparameter acuan ($W = 24$ jam)],
  placement: top,
  table(
    columns: 2,
    align: left,
    stroke: none,
    table.hline(),
    table.header([*Hyperparameter*], [*Nilai Acuan*]),
    table.hline(stroke: 0.5pt),
    [Panjang jendela ($W$)], [24 jam],
    [Horizon ($H$)], [1 jam],
    [Ukuran hidden], [ANN 32, RNN 68, LSTM 33, GRU 38],
    [Jumlah lapisan], [ANN 2, Rekuren 1],
    [Fitur input], [`semua` (7 fitur)],
    [Optimizer], [Adam, $lr = 10^(-3)$],
    [Batch size], [64],
    [Dropout], [0.0],
    [Fungsi loss latih], [MSE],
    [Early stopping], [Patience 30, maks. 300 epoch],
    table.hline(),
  ),
) <tab:acuan>

== Protokol Ablasi dan Evaluasi
Satu hyperparameter divariasikan per eksperimen; 3 seed per konfigurasi. Metrik: MAE, RMSE, MAPE pada skala asli. Konfigurasi terbaik dipilih berdasarkan val MAE. Faktor B pilihan beserta hipotesis:
- *B1 Batch Size* (16, 64, 256): Batch kecil memberikan regularisasi stokastik lebih kuat, diperkirakan menurunkan val MAE.
- *B4 Fungsi Loss* (MSE, MAE, Huber): Loss L1/Huber langsung mengoptimalkan MAE, diperkirakan menghasilkan test MAE lebih rendah.

Perangkat: AMD Ryzen 7 7730U (8 core), PyTorch CPU. Total 348 run dalam 83 menit.

= Hasil

== Baseline Naif dan Konfigurasi Acuan
@tab:naif menunjukkan perbandingan baseline naif dan konfigurasi acuan. Naif terbaik pada minggu test adalah "minggu lalu" (MAE 467.6), mencerminkan siklus mingguan kuat trafik komersial Duomo. Namun saat Natal, baseline "minggu lalu" runtuh (MAE 1426.2, MAPE 96.8%) karena anomali perilaku sosial. Seluruh model pembelajaran mendalam konfigurasi acuan melampaui semua baseline naif.

#figure(
  caption: [Baseline naif dan konfigurasi acuan di Duomo ($H = 1$ jam, rata-rata $plus.minus$ std, 3 seed)],
  placement: top,
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model / Metode*], [*Val MAE*], [*Test MAE*], [*Natal MAE*]),
    table.hline(stroke: 0.5pt),
    [Naif: Nilai Terakhir], [662.6], [600.9], [431.2],
    [Naif: Kemarin], [595.3], [552.2], [921.5],
    [Naif: Minggu Lalu], [492.8], [467.6], [1426.2],
    table.hline(stroke: 0.3pt),
    [ANN (Acuan)], [$248.9 plus.minus 8.9$], [$309.9 plus.minus 5.9$], [$680.1 plus.minus 52.0$],
    [RNN (Acuan)], [$206.1 plus.minus 7.4$], [$229.6 plus.minus 5.1$], [$421.2 plus.minus 2.4$],
    [LSTM (Acuan)], [$230.5 plus.minus 12.2$], [$273.6 plus.minus 25.4$], [$586.0 plus.minus 56.8$],
    [GRU (Acuan)], [$207.5 plus.minus 3.8$], [$237.2 plus.minus 9.3$], [$468.2 plus.minus 21.8$],
    table.hline(),
  ),
) <tab:naif>

== Ablasi Wajib (A1 – A5)

=== A1. Panjang Jendela ($W in {6, 12, 24, 48, 168}$ jam)
@fig:a1 menunjukkan perilaku berbeda antara ANN dan model rekuren. ANN mengalami degradasi signifikan pada $W=168$ karena dimensi input $(168 times 7 = 1176)$ menyebabkan pembengkakan parameter hingga 38.753 unit, memicu overfitting pada 672 sampel latih. Model rekuren (parameter konstan terhadap W) memperoleh keuntungan dari konteks yang lebih panjang hingga $W=48$ jam, sedangkan $W=168$ tidak lagi memberikan manfaat tambahan.

#figure(
  image("../hasil/A1_W.png", width: 100%),
  caption: [A1: Pengaruh panjang jendela historis $W$ terhadap Val MAE dan Test MAE.],
  placement: top,
) <fig:a1>

=== A2. Ukuran Hidden ($in {16, 32, 64, 128}$)
@fig:a2 menunjukkan bahwa model yang lebih besar tidak selalu lebih baik. LSTM dengan hidden 128 memiliki ~4× parameter RNN dengan hidden 128, namun performa validasinya tidak proporsional. GRU hidden 128 memberikan penurunan val MAE paling signifikan (dari 207.5 ke 188.7).

#figure(
  image("../hasil/A2_hidden.png", width: 100%),
  caption: [A2: Pengaruh ukuran hidden state terhadap val MAE dan test MAE.],
  placement: top,
) <fig:a2>

=== A3. Kedalaman Lapisan ($in {1, 2, 3}$)
@fig:a3 membuktikan bahwa lapisan rekuren bertumpuk secara konsisten mendegradasi kinerja. Khususnya LSTM 3 lapis mengalami lonjakan val MAE drastis ($322.1 plus.minus$). Kolom `epoch` mengkonfirmasi model lebih dalam mencapai early stopping pada epoch lebih awal, tanda overfitting.

#figure(
  image("../hasil/A3_layers.png", width: 100%),
  caption: [A3: Pengaruh kedalaman lapisan terhadap performa.],
  placement: top,
) <fig:a3>

=== A4. Learning Rate ($in {10^(-4), 10^(-3), 10^(-2)}$)
@fig:a4 mengkonfirmasi bahwa $lr = 10^(-4)$ menyebabkan seluruh model terhenti di batas 300 epoch (belum konvergen). Nilai $lr = 10^(-2)$ menghasilkan konvergensi cepat (~62–68 epoch) namun dengan variansi antar-seed lebih besar. $lr = 10^(-3)$ memberikan keseimbangan optimal untuk sebagian besar model.

#figure(
  image("../hasil/A4_lr.png", width: 100%),
  caption: [A4: Pengaruh learning rate terhadap stabilitas konvergensi.],
  placement: top,
) <fig:a4>

=== A5. Kelompok Fitur
@fig:a5 mengungkap temuan kunci: fitur waktu trigonometrik memberikan penurunan MAE paling drastis. Pada RNN, `net+waktu` (val MAE 223.0) jauh mengungguli `net` saja (245.2) namun setara dengan `semua` (206.1) yang juga memasukkan call dan sms. Ini menunjukkan informasi call/sms berkontribusi tambahan di atas fitur waktu.

#figure(
  image("../hasil/A5_fitur.png", width: 100%),
  caption: [A5: Pengaruh pemilihan kelompok fitur terhadap akurasi prediksi.],
  placement: top,
) <fig:a5>

== Ablasi Pilihan (B1 dan B4)

=== B1. Batch Size ($in {16, 64, 256}$)
@fig:b1 membuktikan hipotesis: batch 16 unggul secara konsisten pada seluruh model. Batch 256 menghasilkan val MAE lebih tinggi karena estimasi gradien yang terlalu mulus menginduksi konvergensi ke minimum tajam. Efek paling terasa pada ANN.

#figure(
  image("../hasil/B1_batch.png", width: 100%),
  caption: [B1: Pengaruh batch size terhadap val MAE dan test MAE.],
  placement: top,
) <fig:b1>

=== B4. Fungsi Loss (MSE, MAE, Huber)
@fig:b4 membuktikan hipotesis sebagian: loss MAE tidak selalu menghasilkan test MAE terbaik (ada trade-off stabilitas). Untuk model rekuren, Huber loss memberikan kompromi terbaik — lebih stabil dari MAE loss namun lebih akurat dari MSE loss pada metrik MAE evaluasi akhir.

#figure(
  image("../hasil/B4_loss.png", width: 100%),
  caption: [B4: Pengaruh fungsi loss terhadap akurasi prediksi.],
  placement: top,
) <fig:b4>

== Konfigurasi Terbaik dan Perbandingan dengan Acuan
Berdasarkan val MAE terendah per faktor ablasi, konfigurasi terbaik per model adalah: ANN ($W=6$, hidden 128, layers 1, fitur `net+waktu`, batch 16, loss `mae`); RNN ($W=48$, batch 16); LSTM ($W=6$, hidden 128, $lr=0.01$, batch 16); GRU ($W=48$, hidden 128, layers 2, batch 16).

@tab:terbaik dan @fig:c menunjukkan peningkatan test MAE yang signifikan pada seluruh model. ANN memperoleh penurunan terbesar (309.9 → 257.4, −16.9%). RNN dan GRU mempertahankan urutan terbaik.

#figure(
  caption: [Konfigurasi acuan vs terbaik di Duomo (rata-rata $plus.minus$ std, 3 seed)],
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Val MAE*\ Acuan], [*Val MAE*\ Terbaik], [*Test MAE*\ Acuan], [*Test MAE*\ Terbaik]),
    table.hline(stroke: 0.5pt),
    [ANN],  [$248.9 plus.minus 8.9$],  [$bold(219.2 plus.minus 3.0)$],  [$309.9 plus.minus 5.9$],  [$bold(257.4 plus.minus 8.0)$],
    [RNN],  [$206.1 plus.minus 7.4$],  [$bold(209.5 plus.minus 3.2)$],  [$229.6 plus.minus 5.1$],  [$bold(226.3 plus.minus 4.9)$],
    [LSTM], [$230.5 plus.minus 12.2$], [$241.1 plus.minus 4.9$],        [$273.6 plus.minus 25.4$], [$bold(246.7 plus.minus 12.0)$],
    [GRU],  [$207.5 plus.minus 3.8$],  [$227.4 plus.minus 7.3$],        [$237.2 plus.minus 9.3$],  [$bold(230.2 plus.minus 8.5)$],
    table.hline(),
  ),
) <tab:terbaik>

#figure(
  image("../hasil/C_terbaik_vs_acuan.png", width: 95%),
  caption: [Perbandingan test MAE acuan vs terbaik di Duomo terhadap baseline naif.],
  placement: top,
) <fig:c>

== Uji Generalisasi di Universita Bocconi (Kode 2)
@fig:d_pola memperlihatkan perbedaan karakteristik trafik: Duomo memiliki volume 4.000–8.000 transaksi/jam sedangkan Bocconi 400–1.200 transaksi/jam dengan puncak tajam pada jam perkuliahan.

#figure(
  image("../hasil/D_pola_lokasi.png", width: 100%),
  caption: [Profil trafik internet per jam: Duomo (atas) vs Universita Bocconi (bawah).],
  placement: top,
) <fig:d_pola>

@tab:d menunjukkan bahwa GRU terbaik di Bocconi (test MAE $519.3 plus.minus 16.7$) lebih unggul dari GRU acuan ($608.5 plus.minus 127.4$), membuktikan transferabilitas konfigurasi. LSTM menunjukkan degradasi besar pada test Bocconi (849.7), kemungkinan karena pola trafik kampus berbeda dari distribusi pelatihan Duomo.

#figure(
  caption: [Generalisasi ke Bocconi: acuan vs terbaik (test MAE rata-rata $plus.minus$ std)],
  placement: top,
  table(
    columns: 3,
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Test MAE Acuan*], [*Test MAE Terbaik*]),
    table.hline(stroke: 0.5pt),
    [ANN],  [$645.5 plus.minus 57.0$],   [$581.5 plus.minus 104.0$],
    [RNN],  [$754.2 plus.minus 85.7$],   [$587.2 plus.minus 75.5$],
    [LSTM], [$682.1 plus.minus 180.1$],  [$849.7 plus.minus 21.5$],
    [GRU],  [$bold(608.5 plus.minus 127.4)$], [$bold(519.3 plus.minus 16.7)$],
    table.hline(),
  ),
) <tab:d>

= Pembahasan

*Superioritas Model Rekuren*: ANN menderita kelemahan fundamental—meratakan struktur waktu menjadi vektor besar—sedangkan model rekuren mengekstrak dinamika temporal secara bertahap dengan parameter ringkas.

*Faktor Terpenting*: Kelompok fitur (A5) dan panjang jendela (A1) memberikan dampak terbesar. Fitur waktu trigonometrik mengkodekan siklus sirkadian secara eksplisit sehingga model tidak perlu belajar dari nol.

*Over-parametrisasi Berbahaya*: Pada 672 jam data latih, model dalam (lapisan > 1) mengalami overfitting cepat. Batch kecil (16) memberikan regularisasi implisit yang efektif sebagai pengganti dropout.

*Transferabilitas Konfigurasi*: GRU dan RNN terbaik berhasil digeneralisasikan ke Bocconi; LSTM terbaik tidak, kemungkinan karena konfigurasi $lr=0.01$ optimal untuk distribusi Duomo tetapi tidak stabil pada skala trafik berbeda.

*Analisis Robustness Hari Libur*: Seluruh model mengalami lonjakan nilai MAE yang signifikan pada minggu Natal. Hal ini terjadi akibat adanya *distribution shift*, di mana pola konsumsi data masyarakat berubah secara drastis dari rutinitas harian biasa menjadi aktivitas liburan, sehingga pola histori yang dipelajari model selama masa pelatihan menjadi kurang relevan.

*Rekomendasi Praktis Implementasi*: Untuk skenario penyebaran di industri telekomunikasi nyata, model **GRU 1-layer** dengan fitur waktu trigonometrik dan *batch size* 16 merupakan opsi paling ideal. Arsitektur ini memberikan keseimbangan optimal antara akurasi prediksi, efisiensi penggunaan memori, serta waktu komputasi yang lebih cepat dibandingkan LSTM.

*Keterbatasan*: Studi ini terbatas pada satu kota (Milan), satu musim (musim dingin 2013), dan pembagian waktu tunggal. Evaluasi multi-kota dan lintas musim diperlukan untuk validasi yang lebih umum.

= Kesimpulan
Studi ablasi sistematis terhadap 348 run dengan 7 faktor hyperparameter membuktikan bahwa model rekuren (RNN dan GRU) secara konsisten mengungguli ANN untuk prediksi trafik seluler per jam. Pada konfigurasi acuan, RNN mencapai test MAE $229.6 plus.minus 5.1$ dibandingkan ANN $309.9 plus.minus 5.9$. Setelah optimasi, konfigurasi terbaik berhasil menurunkan test MAE menjadi RNN $226.3 plus.minus 4.9$ dan GRU $230.2 plus.minus 8.5$. Fitur waktu trigonometrik terbukti sebagai faktor paling berpengaruh; penumpukan lapisan rekuren kontraproduktif pada dataset kecil. Konfigurasi GRU terbaik berhasil digeneralisasikan ke Universita Bocconi dengan test MAE $519.3 plus.minus 16.7$, membuktikan transferabilitas representasi yang dipelajari.

#heading(numbering: none)[Kontribusi Anggota]
#table(
  columns: (auto, 1fr),
  stroke: none,
  [*Isabella Silalahi*], [Setup repository dan CI GitHub, implementasi dan eksekusi notebook Bagian 0 dan A1–A3, analisis hyperparameter arsitektur dan kedalaman lapisan.],
  [*Dwi Yanti Maharani*], [Ablasi Bagian A4–A5 (learning rate dan fitur), Bagian B1 (batch size) dan B4 (loss function), penyusunan visualisasi dan draf bagian Metode dan Hasil.],
  [*Lusitania Fachria Salsabila*], [Identifikasi lokasi kelompok Bocconi (kode 2), eksekusi Bagian C (konfigurasi terbaik) dan Bagian D (generalisasi), penyusunan laporan IEEE dan analisis Pembahasan.],
)

#heading(numbering: none)[Pernyataan Penggunaan AI]
Asisten kecerdasan buatan berbasis model bahasa (Antigravity AI) digunakan sebagai alat bantu pemrograman pasangan dalam verifikasi sintaksis kode Python, automasi eksekusi notebook Jupyter via `nbclient`, kompilasi dokumen Typst, dan perapian redaksional bahasa akademis. Seluruh data eksperimental, angka numerik, dan grafik berasal murni dari eksekusi notebook pada perangkat lokal (348 run, 83 menit).

#heading(numbering: none)[Kode dan Data]
Kode sumber lengkap, notebook Jupyter beserta seluruh output sel, dan berkas CSV/PNG hasil tersedia di: \
#link("https://github.com/bellatolping-oss/tugas-deep-learning")
