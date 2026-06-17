import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    """
    Melakukan preprocessing pada gambar sebelum inferensi model.
    """
    # 1. Resize gambar sesuai dimensi input CNN
    resized = cv2.resize(image, target_size)
    
    # 2. Konversi format BGR (OpenCV) ke RGB
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalisasi nilai piksel (0 - 1)
    normalized_image = rgb_image / 255.0
    
    # 4. Ubah ke array numpy dan tambahkan dimensi batch (1, 224, 224, 3)
    input_tensor = np.expand_dims(normalized_image, axis=0)
    
    return input_tensor
