import os
import cv2
import numpy as np

# --- Import MediaPipe Tasks API (v0.10+) dengan fallback ke Haar Cascades ---
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Path ke model face_landmarker.task (didownload sekali)
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'face_landmarker.task')
    _MODEL_PATH = os.path.abspath(_MODEL_PATH)
    _MP_AVAILABLE = os.path.exists(_MODEL_PATH)

    if _MP_AVAILABLE:
        print(f"[roi_detection] MediaPipe FaceLandmarker aktif. Model: {_MODEL_PATH}")
    else:
        print(f"[roi_detection] PERINGATAN: model face_landmarker.task tidak ditemukan di {_MODEL_PATH}. Fallback ke Haar Cascades.")
except ImportError:
    _MP_AVAILABLE = False
    print("[roi_detection] PERINGATAN: MediaPipe tidak terinstall. Fallback ke Haar Cascades.")


# ---
# Indeks Landmark FaceMesh untuk area kelopak mata bawah (konjungtiva palpebral)
# Referensi: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ---

# Kontur kelopak mata bawah KIRI (0-jika dilihat dari depan)
_LEFT_EYE_LOWER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
# Kontur kelopak mata bawah KANAN
_RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]

# Titik iris kiri & kanan (tersedia di model dengan output_face_blendshapes=True atau model lengkap)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473


def _landmarks_to_points(landmarks, indices, img_w, img_h):
    """Konversi landmark ke koordinat piksel."""
    pts = []
    for idx in indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    return pts


def _extract_roi_from_eye_lower(image, eye_lower_pts, padding=0.5):
    """
    Ekstrak bounding box ROI konjungtiva dari titik-titik kelopak bawah.
    Area konjungtiva terletak di separuh bawah mata, diperluas ke arah bawah.
    """
    if not eye_lower_pts:
        return None

    h_img, w_img = image.shape[:2]
    pts_arr = np.array(eye_lower_pts)
    x_min, y_min = pts_arr.min(axis=0)
    x_max, y_max = pts_arr.max(axis=0)

    eye_h = max(y_max - y_min, 1)
    eye_w = max(x_max - x_min, 1)

    # Area konjungtiva = kelopak bawah; perluas ke bawah
    y_start = int(y_min + eye_h * 0.5)
    y_end = min(int(y_max + eye_h * padding), h_img)
    x_start = max(int(x_min - eye_w * 0.05), 0)
    x_end = min(int(x_max + eye_w * 0.05), w_img)

    roi = image[y_start:y_end, x_start:x_end]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None
    return roi


def extract_conjunctiva_roi(image):
    """
    Fungsi utama: Ekstrak ROI konjungtiva.
    
    Pipeline:
      1. MediaPipe Tasks FaceLandmarker (jika model tersedia)
      2. Fallback: OpenCV Haar Cascades
    
    Mengembalikan numpy array (BGR) atau None jika gagal.
    """
    if _MP_AVAILABLE:
        roi = _extract_with_facelandmarker(image)
        if roi is not None:
            return roi
        # Jika FaceLandmarker gagal (tidak ada wajah), coba Haar
        print("[roi_detection] FaceLandmarker tidak dapat mendeteksi wajah. Mencoba Haar Cascades...")

    return _extract_with_haar_cascade(image)


def _extract_with_facelandmarker(image):
    """
    Implementasi utama menggunakan MediaPipe FaceLandmarker (PipeMask/FaceMesh penuh).
    Diinisialisasi sekali per panggilan menggunakan konteks 'with'.
    """
    h_img, w_img = image.shape[:2]

    # Konversi ke RGB (MediaPipe menggunakan SRGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Konfigurasi FaceLandmarker
    base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,        # Tidak perlu blend shapes
        output_facial_transformation_matrixes=False,
        num_faces=1,                          # Fokus pada satu wajah
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE  # Mode statis
    )

    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    landmarks = result.face_landmarks[0]  # Ambil wajah pertama

    # Ambil titik kelopak bawah kiri & kanan
    left_lower = _landmarks_to_points(landmarks, _LEFT_EYE_LOWER, w_img, h_img)
    right_lower = _landmarks_to_points(landmarks, _RIGHT_EYE_LOWER, w_img, h_img)

    # Prioritas: mata kiri, fallback ke kanan
    roi = _extract_roi_from_eye_lower(image, left_lower, padding=0.5)
    if roi is None:
        roi = _extract_roi_from_eye_lower(image, right_lower, padding=0.5)

    return roi


def _extract_with_haar_cascade(image):
    """Fallback: Deteksi area mata dengan Haar Cascades (OpenCV bawaan)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(eyes) == 0:
        return None

    eyes = sorted(eyes, key=lambda x: x[2] * x[3], reverse=True)
    (ex, ey, ew, eh) = eyes[0]
    h_img, w_img = image.shape[:2]

    margin_x = int(ew * 0.1)
    y_lower_start = int(ey + eh * 0.6)
    y_lower_end = min(ey + eh + int(eh * 0.2), h_img)
    x_start = max(ex + margin_x, 0)
    x_end = min(ex + ew - margin_x, w_img)

    roi = image[y_lower_start:y_lower_end, x_start:x_end]
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        return None
    return roi
