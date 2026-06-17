import os
import cv2
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

keras_model = None
yolo_model = None

# Daftar kemungkinan lokasi file model YOLO (.pt)
POSSIBLE_YOLO_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'yolo11_anemia.pt'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'best.pt'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'best.pt'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolo11n-cls.pt'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'yolo11n.pt')
]

# Daftar kemungkinan lokasi file model CNN (.h5)
POSSIBLE_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'cnn_model.h5'),
    os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'anaemicvsnonanaemic.h5'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'anemia_cnn_model.h5'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'anemia_detection_model.h5')
]

def load_yolo_model():
    """
    Memuat model YOLOv11 (.pt) dari lokasi yang tersedia.
    """
    global yolo_model
    if not YOLO_AVAILABLE:
        print("Peringatan: Ultralytics/YOLO belum terinstal.")
        return False

    for path in POSSIBLE_YOLO_PATHS:
        if os.path.exists(path):
            try:
                print(f"Mencoba memuat model YOLO dari: {path}")
                yolo_model = YOLO(path)
                print(f"Model YOLO berhasil dimuat dari: {path}")
                return True
            except Exception as e:
                print(f"Gagal memuat model YOLO dari {path}: {e}")
    return False

def load_system_model():
    """
    Memuat model CNN (.h5) dari lokasi yang tersedia.
    """
    global keras_model
    if not TF_AVAILABLE:
        print("Peringatan: TensorFlow belum terinstal.")
        return False

    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            try:
                print(f"Mencoba memuat model Keras dari: {path}")
                keras_model = tf.keras.models.load_model(path)
                print(f"Model Keras berhasil dimuat dari: {path}")
                return True
            except Exception as e:
                print(f"Gagal memuat model Keras dari {path}: {e}")
    return False

def predict_image(preprocessed_image):
    """
    Melakukan inferensi dari gambar ROI konjungtiva yang telah diproses.
    Mendukung model YOLOv11 (.pt) dan fallback ke Keras CNN (.h5) / Dummy logic.
    """
    global yolo_model, keras_model

    # 1. Coba gunakan YOLOv11 jika terinstal dan model ditemukan
    if YOLO_AVAILABLE:
        if yolo_model is None:
            load_yolo_model()
        
        if yolo_model is not None:
            try:
                # Rekonstruksi gambar BGR dari tensor preprocessed (untuk YOLO)
                rgb_img = (preprocessed_image[0] * 255.0).astype(np.uint8)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                results = yolo_model(bgr_img, verbose=False)
                result = results[0]

                # Cek jika model klasifikasi (YOLO-cls)
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top1_idx = int(probs.top1)
                    confidence = float(probs.top1conf)
                    
                    # Pemetaan label
                    class_name = yolo_model.names[top1_idx]
                    pred_label = "Anemia" if "anemi" in class_name.lower() else "Normal"
                    pred_class = 1 if pred_label == "Anemia" else 0

                    return {
                        "prediction_class": pred_class,
                        "prediction_label": pred_label,
                        "confidence": round(confidence, 4),
                        "model_type": f"YOLOv11-Classification ({class_name})"
                    }
                
                # Cek jika model deteksi (YOLO-detect)
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Ambil deteksi dengan confidence tertinggi
                    best_box = None
                    best_conf = -1.0
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf > best_conf:
                            best_conf = conf
                            best_box = box
                    
                    class_idx = int(best_box.cls[0])
                    class_name = yolo_model.names[class_idx]
                    pred_label = "Anemia" if "anemi" in class_name.lower() else "Normal"
                    pred_class = 1 if pred_label == "Anemia" else 0

                    return {
                        "prediction_class": pred_class,
                        "prediction_label": pred_label,
                        "confidence": round(best_conf, 4),
                        "model_type": f"YOLOv11-Detection ({class_name})"
                    }
            except Exception as e:
                print("Gagal inferensi YOLO:", e)

    # 2. Fallback ke Keras CNN (.h5) jika YOLO tidak tersedia/gagal
    if keras_model is None:
        load_system_model()

    if keras_model is not None:
        try:
            prediction = keras_model.predict(preprocessed_image)
            score = float(prediction[0][0])
            
            pred_class = 1 if score > 0.5 else 0
            confidence = score if pred_class == 1 else (1 - score)
            
            return {
                "prediction_class": pred_class,
                "prediction_label": "Anemia" if pred_class == 1 else "Normal",
                "confidence": round(confidence, 4),
                "model_type": "Keras-CNN"
            }
        except Exception as e:
            print("Gagal inferensi Keras:", e)

    # 3. Fallback terakhir ke Dummy logic
    print("Menggunakan model fallback dummy.")
    confidence = float(np.random.uniform(0.6, 0.99))
    pred_class = 1 if np.random.rand() > 0.5 else 0
    return {
        "prediction_class": pred_class,
        "prediction_label": "Anemia" if pred_class == 1 else "Normal",
        "confidence": round(confidence, 4),
        "model_type": "Dummy Fallback"
    }

