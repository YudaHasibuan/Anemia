# Deteksi Anemia Berbasis Web (Non-Invasive Web-Based Anemia Detection)

Aplikasi deteksi anemia non-invasif berbasis web ini memanfaatkan teknologi **Deep Learning** tingkat lanjut menggunakan **YOLOv11** dan **Computer Vision** untuk mendeteksi risiko anemia melalui analisis warna konjungtiva (kelopak mata bagian bawah). Aplikasi ini dirancang agar dapat diakses dengan mudah secara real-time menggunakan kamera web (webcam) atau perangkat seluler (smartphone).

---

## 🚀 Fitur Utama

1. **Integrasi YOLOv11 & CNN**:
   - Mendukung model **YOLOv11** (format `.pt` baik tipe klasifikasi *YOLO11-cls* maupun deteksi *YOLO11-detect*).
   - Memiliki fallback otomatis ke model **Convolutional Neural Network (CNN)** (format `.h5` berbasis Keras/TensorFlow) jika model YOLO tidak terdeteksi atau pustaka belum siap.
2. **Deteksi Landmark Wajah & Mata Real-Time**: 
   - Menggunakan **MediaPipe FaceLandmarker** untuk melacak posisi mata secara presisi.
   - Menyediakan fallback otomatis menggunakan **OpenCV Haar Cascades** jika modul MediaPipe tidak terinstal atau tidak mendeteksi wajah dengan baik.
3. **Ekstraksi ROI (Region of Interest) Konjungtiva**:
   - Memotong area kelopak mata bawah secara otomatis berdasarkan titik landmark kelopak mata bagian bawah (*conjunctiva palpebral*).
4. **Antarmuka Responsif & PWA (Progressive Web App)**:
   - Dilengkapi dengan *Service Worker* (`sw.js`) dan manifestasi aplikasi (`manifest.json`) sehingga mendukung mode instalasi di perangkat mobile (PWA).
   - UI interaktif dengan umpan kamera real-time dan visualisasi hasil deteksi yang modern.
5. **Halaman Evaluasi Model**:
   - Menyediakan antarmuka visual untuk menampilkan laporan kinerja evaluasi model (akurasi, loss, confusion matrix, dll.).

---

## 📁 Struktur Proyek

```text
Anemia/
│
├── anemia/                     # Direktori Utama Kode Flask
│   ├── model/                  # Direktori Model Pre-Trained
│   │   ├── face_landmarker.task      # Model MediaPipe Landmark Wajah
│   │   ├── yolo11_anemia.pt          # Model YOLOv11 Custom (Diutamakan)
│   │   └── anaemicvsnonanaemic.h5     # Model CNN Ringan (Fallback)
│   │
│   ├── static/                 # Aset Statis (CSS, JS, Gambar, PWA)
│   │   ├── css/                # Styling CSS halaman web
│   │   ├── js/                 # Logika Kamera & API Request JavaScript
│   │   ├── manifest.json       # Konfigurasi PWA Manifest
│   │   └── sw.js               # Service Worker PWA
│   │
│   ├── templates/              # Template HTML (Jinja2)
│   │   ├── index.html          # Halaman Beranda (Landing Page)
│   │   ├── detect.html         # Antarmuka Kamera Real-Time
│   │   ├── result.html         # Tampilan Hasil Deteksi
│   │   └── evaluation.html     # Laporan Evaluasi Model (YOLOv11)
│   │
│   ├── utils/                  # Modul Pendukung Python
│   │   ├── roi_detection.py    # Logika deteksi mata & ekstraksi kelopak bawah
│   │   ├── preprocessing.py    # Preprocessing gambar input model (224x224, normalisasi)
│   │   └── predict.py          # Logika inferensi YOLOv11 & Keras CNN
│   │
│   ├── app.py                  # Entry Point Server Flask
│   └── requirements.txt        # Dependensi Python (termasuk ultralytics)
│
├── notebook1122b3ff41.ipynb    # Jupyter Notebook pelatihan model CNN
├── anemia_cnn_model.h5         # Model CNN Utama (Hasil Pelatihan - Diabaikan oleh Git karena >100MB)
└── anemia_detection_model.h5   # Model CNN Alternatif (Hasil Pelatihan - Diabaikan oleh Git karena >100MB)
```

---

## 🛠️ Panduan Instalasi dan Penggunaan

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi secara lokal di komputer Anda:

### 1. Prasyarat
Pastikan komputer Anda sudah terinstal:
- **Python 3.9 - 3.12**
- **Pip** (Manajer paket Python)
- **Git**

### 2. Kloning Repositori
Klon repositori ini dari GitHub:
```bash
git clone https://github.com/YudaHasibuan/Anemia.git
cd Anemia
```

### 3. Setup Virtual Environment (Rekomendasi)
Buat dan aktifkan virtual environment untuk menjaga dependensi tetap bersih:

- **Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Instal Dependensi
Instal pustaka Python yang diperlukan (termasuk `ultralytics` untuk YOLOv11):
```bash
pip install -r requirements.txt
```

### 5. Setup Model YOLOv11 / CNN
Sistem secara otomatis mendeteksi model yang tersedia. Letakkan file model Anda di salah satu direktori berikut:

#### A. Opsi Utama: YOLOv11 (`.pt`)
Letakkan model YOLO Anda dengan nama file dan lokasi berikut (dipindai berdasarkan prioritas):
1. `anemia/model/yolo11_anemia.pt`
2. `anemia/model/best.pt`
3. `best.pt` (di root proyek)
4. `yolo11n-cls.pt` (di root proyek)

#### B. Opsi Fallback: Keras CNN (`.h5`)
Jika model YOLO tidak ditemukan atau library Ultralytics tidak terpasang, sistem akan memindai model Keras `.h5`:
1. `anemia/model/cnn_model.h5`
2. `anemia/model/anaemicvsnonanaemic.h5`
3. `anemia_cnn_model.h5` (di root proyek)

*Jika tidak ada model valid yang terdeteksi, aplikasi akan berjalan menggunakan **dummy fallback logic** untuk keperluan pengetesan alur web.*

### 6. Menjalankan Aplikasi Web
Masuk ke folder `anemia` dan jalankan file `app.py`:
```bash
cd anemia
python app.py
```

Setelah aplikasi berjalan, buka peramban web (browser) Anda dan akses alamat berikut:
👉 **`http://localhost:5002`**

---

## ⚙️ Cara Kerja Deteksi

1. **Input Kamera**: Kamera menangkap wajah pengguna melalui antarmuka web HTML5.
2. **Pelacakan Wajah**: Aplikasi mengirim frame ke backend Flask, di mana **MediaPipe** melacak landmark wajah secara detail.
3. **Ekstraksi Kelopak Mata**: Titik landmark kelopak mata bawah kiri/kanan diproses untuk mendefinisikan *bounding box* konjungtiva secara otomatis.
4. **Preprocessing**: Gambar kelopak mata dipotong (crop) dan diubah ukurannya menjadi `224x224` piksel.
5. **Inferensi YOLOv11**:
   - Jika model berupa **YOLO11-cls** (klasifikasi), gambar diinferensi langsung untuk mengeluarkan prediksi `Anemia` atau `Normal` beserta skor keyakinan (*confidence*).
   - Jika model berupa **YOLO11-detect** (deteksi objek), sistem mengambil objek terdeteksi dengan skor keyakinan tertinggi untuk menentukan kelasnya.
