import os
import cv2
import numpy as np
import functools
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, flash
from utils.roi_detection import extract_conjunctiva_roi
from utils.preprocessing import preprocess_image
from utils.predict import predict_image

app = Flask(__name__)
app.secret_key = 'super_secret_yolov11_anemia_dashboard_key_2026'

# Konfigurasi folder penyimpanan opsional
UPLOAD_FOLDER = os.path.join('static', 'images', 'temp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js')

@app.route('/')
def index():
    """Halaman Beranda."""
    return render_template('index.html')

@app.route('/detect')
def detect():
    """Halaman Antarmuka Kamera untuk Deteksi."""
    return render_template('detect.html')

@app.route('/result')
def result():
    """Halaman Hasil (Hanya template antarmuka kosong, data diatur melalui JS)."""
    return render_template('result.html')

@app.route('/evaluation')
def evaluation():
    """Halaman Laporan Evaluasi dan Validasi Model."""
    return render_template('evaluation.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint API untuk menerima gambar yang diunggah oleh pengguna 
    dan mengembalikan hasil inferensi.
    """
    if 'image' not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan dalam permintaan."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File kosong."}), 400

    try:
        # Membaca byte aliran dari memori langsung, tidak menyimpannya ke disk (untuk privasi)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
             return jsonify({"error": "Gagal mendekode array byte gambar."}), 400

        # Langkah 1: Ekstraksi ROI Konjungtiva
        roi = extract_conjunctiva_roi(image)
        if roi is None:
            return jsonify({
                "error": "Tidak dapat mendeteksi mata/wajah Anda secara jelas. Pastikan sistem dapat melihat area mata bagian bawah secara terang."
            }), 400

        # Langkah 2: Preprocessing ROI (Resize, Normalisasi, dll)
        preprocessed = preprocess_image(roi, target_size=(224, 224))

        # Langkah 3: Inferensi dengan Model .h5
        result = predict_image(preprocessed)

        # Kembalikan response JSON
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error pada /predict API: {e}")
        return jsonify({"error": "Terjadi kesalahan internal pada server. Silakan coba lagi."}), 500

# Inisialisasi Haar Cascade sekali saat startup (hemat memori)
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """
    Endpoint real-time untuk deteksi wajah + mata menggunakan OpenCV Haar Cascades.
    Menerima frame JPEG dari browser, mengembalikan koordinat kotak deteksi.
    """
    if 'frame' not in request.files:
        return jsonify({'faces': [], 'eyes': []}), 200

    try:
        file_bytes = np.frombuffer(request.files['frame'].read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'faces': [], 'eyes': []}), 200

        h_img, w_img = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Tingkatkan kontras

        # Deteksi wajah
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        result_faces = []
        result_eyes  = []

        for (fx, fy, fw, fh) in faces:
            result_faces.append({
                'x': int(fx / w_img * 100),   # Simpan sebagai % agar responsif
                'y': int(fy / h_img * 100),
                'w': int(fw / w_img * 100),
                'h': int(fh / h_img * 100)
            })

            # Deteksi mata di dalam ROI wajah
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = _eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
            )
            # Ambil maks 2 mata teratas
            eyes_sorted = sorted(eyes, key=lambda e: e[3] * e[2], reverse=True)[:2] if len(eyes) else []
            for (ex, ey, ew, eh) in eyes_sorted:
                # Koordinat absolut di frame asli (dalam %)
                abs_ex = fx + ex
                abs_ey = fy + ey
                result_eyes.append({
                    'x':  int(abs_ex / w_img * 100),
                    'y':  int(abs_ey / h_img * 100),
                    'w':  int(ew / w_img * 100),
                    'h':  int(eh / h_img * 100),
                    # Area konjungtiva: separuh bawah mata
                    'cy': int((abs_ey + eh * 0.6) / h_img * 100),
                    'ch': int((eh * 0.5) / h_img * 100),
                })

        return jsonify({
            'faces': result_faces,
            'eyes':  result_eyes,
            'detected': len(result_faces) > 0
        })

    except Exception as e:
        print(f"Error /detect_frame: {e}")
        return jsonify({'faces': [], 'eyes': [], 'detected': False}), 200

# Otentikasi dan Dashboard Admin
ADMIN_CREDENTIALS = {
    'username': 'admin',
    'password': 'password123'
}

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if not session.get('logged_in'):
            flash('Silakan login terlebih dahulu untuk mengakses dashboard.', 'danger')
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('admin_dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
            session['logged_in'] = True
            session['username'] = username
            flash('Selamat datang kembali, admin!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Username atau password salah.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah berhasil keluar.', 'success')
    return redirect(url_for('login'))

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    return render_template('admin_dashboard.html', active_page='dashboard')

@app.route('/admin/dataset', methods=['GET', 'POST'])
@login_required
def admin_dataset():
    dataset_dir = os.path.join(app.root_path, 'static', 'images', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Menangani unggahan file baru ke dataset
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Ambil nama file dan simpan ke folder dataset
                filename = file.filename
                file.save(os.path.join(dataset_dir, filename))
                flash(f'Gambar {filename} berhasil ditambahkan ke dataset!', 'success')
                return redirect(url_for('admin_dataset'))
                
    # Memindai file gambar secara dinamis dari folder dataset
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    try:
        all_files = os.listdir(dataset_dir)
        dataset_images = [f for f in all_files if f.lower().endswith(valid_extensions)]
    except Exception:
        dataset_images = []
        
    total_images = len(dataset_images)
    # Tampilkan max 40 gambar pertama di halaman agar tidak lag
    display_images = sorted(dataset_images)[:40]
    
    return render_template('admin_dataset.html', active_page='dataset', images=display_images, total_images=total_images)

@app.route('/admin/history')
@login_required
def admin_history():
    return render_template('admin_history.html', active_page='history')

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if request.method == 'POST':
        new_password = request.form.get('password')
        if new_password:
            ADMIN_CREDENTIALS['password'] = new_password
            flash('Password berhasil diperbarui!', 'success')
        else:
            flash('Password tidak boleh kosong.', 'danger')
    return render_template('admin_settings.html', active_page='settings')
if __name__ == '__main__':
    # Mode debug bisa dinonaktifkan pada tahap produksi
    app.run(debug=True, host='0.0.0.0', port=5002)
