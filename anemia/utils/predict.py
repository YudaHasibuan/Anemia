import os
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

model = None

# Daftar kemungkinan lokasi file model (.h5)
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'cnn_model.h5'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'anaemicvsnonanaemic.h5'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'anemia_cnn_model.h5'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'anemia_detection_model.h5')
]

def load_system_model():
    """
    Memuat model CNN (.h5) dari lokasi yang tersedia.
    """
    global model
    if not TF_AVAILABLE:
        print("Peringatan: TensorFlow belum terinstal. Model akan menggunakan fallback/dummy logic.")
        return

    loaded = False
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            try:
                print(f"Mencoba memuat model dari: {path}")
                model = tf.keras.models.load_model(path)
                print(f"Model berhasil dimuat dari: {path}")
                loaded = True
                break
            except Exception as e:
                print(f"Gagal memuat model dari {path}: {e}")
    
    if not loaded:
        print("Peringatan: Tidak ada file model valid yang ditemukan. Aplikasi akan menggunakan fallback/dummy logic.")

def predict_image(preprocessed_image):
    """
    Melakukan inferensi dari gambar ROI konjungtiva yang telah diproses.
    Menggunakan model .h5 untuk membedakan antara Anemia dan Non-Anemia.
    """
    global model
    if model is None:
        load_system_model()

    # Jika model masih belum tersedia, lakukan sistem fallback "dummy"
    # untuk membantu pengujian alur program
    if model is None:
        print("Menggunakan model fallback dummy.")
        confidence = float(np.random.uniform(0.6, 0.99))
        pred_class = 1 if np.random.rand() > 0.5 else 0
        return {
            "prediction_class": pred_class,
            "prediction_label": "Anemia" if pred_class == 1 else "Normal",
            "confidence": round(confidence, 4)
        }

    # Melakukan inferensi dengan model asli
    try:
        prediction = model.predict(preprocessed_image)
        score = float(prediction[0][0])
        
        # Kelas 0 -> Non-Anemia, Kelas 1 -> Anemia
        pred_class = 1 if score > 0.5 else 0
        confidence = score if pred_class == 1 else (1 - score)
        
        return {
            "prediction_class": pred_class,
            "prediction_label": "Anemia" if pred_class == 1 else "Normal",
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        print("Gagal inferensi:", e)
        return {"error": "Kesalahan inferensi."}

