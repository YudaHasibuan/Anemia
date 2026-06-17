document.addEventListener('DOMContentLoaded', () => {
    // =============================================
    // MODE SWITCHER (Camera / Upload)
    // =============================================
    const tabCamera = document.getElementById('tabCamera');
    const tabUpload = document.getElementById('tabUpload');
    const panelCamera = document.getElementById('panelCamera');
    const panelUpload = document.getElementById('panelUpload');
    const captureBtn = document.getElementById('captureBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const uploadPreview = document.getElementById('uploadPreview');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    
    let selectedFile = null;
    let cameraActive = false;

    // Switch to Camera Tab
    tabCamera.addEventListener('click', () => {
        tabCamera.classList.add('tab-active');
        tabUpload.classList.remove('tab-active');
        panelCamera.classList.remove('hidden');
        panelUpload.classList.add('hidden');
        startCamera();
    });

    // Switch to Upload Tab
    tabUpload.addEventListener('click', () => {
        tabUpload.classList.add('tab-active');
        tabCamera.classList.remove('tab-active');
        panelUpload.classList.remove('hidden');
        panelCamera.classList.add('hidden');
        stopCamera();
    });

    // =============================================
    // CAMERA MODE
    // =============================================
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');

    function startCamera() {
        if (cameraActive) return;
        const constraints = {
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
        };
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(constraints)
                .then((stream) => {
                    video.srcObject = stream;
                    cameraActive = true;
                })
                .catch((err) => {
                    console.error("Kesalahan akses kamera: ", err);
                    alert("Tidak dapat mengakses kamera. Pastikan Anda telah memberikan izin.");
                });
        }
    }

    function stopCamera() {
        if (video && video.srcObject) {
            video.srcObject.getTracks().forEach(t => t.stop());
            video.srcObject = null;
            cameraActive = false;
        }
    }

    // Init camera on load
    startCamera();

    // Capture from camera
    captureBtn.addEventListener('click', () => {
        if (!video.srcObject) { alert("Kamera belum aktif."); return; }
        captureBtn.disabled = true;
        loading.classList.remove('hidden');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
            if (!blob) { sessionStorage.setItem('error', 'Gagal memproses gambar.'); window.location.href = '/result'; return; }
            sendToApi(blob);
        }, 'image/jpeg', 0.95);
    });

    // =============================================
    // UPLOAD MODE
    // =============================================
    const dropZone = document.getElementById('dropZone');

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });
    dropZone.addEventListener('click', () => fileInput.click());

    function handleFileSelection(file) {
        if (!file.type.startsWith('image/')) { alert('Hanya file gambar yang diizinkan.'); return; }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            uploadPreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
            uploadBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Send uploaded image
    uploadBtn.addEventListener('click', () => {
        if (!selectedFile) { alert('Pilih foto terlebih dahulu.'); return; }
        uploadBtn.disabled = true;
        loading.classList.remove('hidden');
        sendToApi(selectedFile);
    });

    // =============================================
    // SHARED API CALL
    // =============================================
    function sendToApi(imageBlob) {
        const formData = new FormData();
        formData.append('image', imageBlob, 'mata_konjungtiva.jpg');
        fetch('/predict', { method: 'POST', body: formData })
            .then(resp => {
                if (!resp.ok) return resp.json().then(d => { throw new Error(d.error || `HTTP ${resp.status}`); });
                return resp.json();
            })
            .then(data => {
                if (data.error) sessionStorage.setItem('error', data.error);
                else {
                    sessionStorage.setItem('prediction', data.prediction_label);
                    sessionStorage.setItem('confidence', data.confidence);
                    sessionStorage.setItem('model_type', data.model_type || 'YOLOv11');
                }
                window.location.href = '/result';
            })
            .catch(err => {
                sessionStorage.setItem('error', err.message || 'Gagal berkomunikasi dengan server.');
                window.location.href = '/result';
            });
    }
});
