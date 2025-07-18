import cv2
import numpy as np
from PIL import Image
import streamlit as st
import logging
from pyzbar import pyzbar

logger = logging.getLogger(__name__)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not installed. Document classification will use simpler methods.")

# Load face detection model
@st.cache_resource
def load_face_detector():
    try:
        # Use OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        logger.error(f"Error loading face detector: {e}")
        return None

def detect_face(image):
    """
    Detect and extract face from passport image
    
    Args:
        image: PIL Image containing passport
        
    Returns:
        face_image: Extracted face as PIL Image or None
        face_coords: Coordinates of detected face or None
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for face detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Load face detector
    face_cascade = load_face_detector()
    if face_cascade is None:
        return None, None
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, None
    
    # Get the largest face (by area)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Extract face region
    face_img = img_array[y:y+h, x:x+w]
    
    # Convert back to PIL Image
    face_pil = Image.fromarray(face_img)
    
    return face_pil, (x, y, w, h)

def verify_face_quality(face_image):
    """
    Verify if the extracted face meets passport photo requirements
    
    Args:
        face_image: PIL Image of the face
        
    Returns:
        is_valid: Boolean indicating if face meets requirements
        issues: List of detected issues
    """
    if face_image is None:
        return False, ["No face detected"]
    
    issues = []
    
    # Convert to numpy array
    face_array = np.array(face_image)
    
    # Check image size
    h, w = face_array.shape[:2]
    if h < 200 or w < 200:
        issues.append("Face image too small (min 200x200 pixels)")
    
    # Check if grayscale or color
    if len(face_array.shape) < 3 or face_array.shape[2] < 3:
        issues.append("Image should be in color")
    
    # Check brightness
    if len(face_array.shape) == 3:
        gray = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_array
    
    brightness = np.mean(gray)
    if brightness < 80:
        issues.append("Image too dark")
    elif brightness > 200:
        issues.append("Image too bright")
    
    # Check contrast
    contrast = np.std(gray)
    if contrast < 40:
        issues.append("Image has low contrast")
    
    # Check blurriness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Face image is blurry")
    
    return len(issues) == 0, issues

def classify_document(image):
    """
    Classify document type based on visual features
    
    Args:
        image: PIL Image of the document
        
    Returns:
        doc_type: Detected document type
        confidence: Confidence score
    """
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow not installed. Using basic document classification. Install TensorFlow for better results: `pip install tensorflow`")
    
    # Simple rule-based classification
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    h, w = img_array.shape[:2]
    aspect_ratio = w / h
    
    # Check for QR codes or barcodes
    barcodes = pyzbar.decode(img_array)
    has_qr = any(b.type == 'QRCODE' for b in barcodes)
    has_barcode = len(barcodes) > 0
    
    # Check for MRZ (Machine Readable Zone)
    # This is a simplified check - in reality, you'd use OCR
    has_mrz = False
    bottom_region = img_array[int(h*0.7):h, :]
    if len(bottom_region.shape) == 3:
        gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2GRAY)
    else:
        gray_bottom = bottom_region
    
    _, binary = cv2.threshold(gray_bottom, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        if w_rect > bottom_region.shape[1] * 0.5 and 10 < h_rect < 50:
            has_mrz = True
            break
    
    # Classify based on features
    if has_mrz and 1.4 < aspect_ratio < 1.6:
        return "Passport", 0.9
    elif has_mrz and aspect_ratio > 1.6:
        return "ID Card", 0.8
    elif has_qr:
        return "Modern Passport/ID", 0.85
    elif has_barcode:
        return "Travel Document", 0.7
    else:
        return "Unknown Document", 0.5

def detect_qr_codes(image):
    """
    Detect and decode QR codes in the image
    
    Args:
        image: PIL Image to scan
        
    Returns:
        qr_data: List of decoded QR code data
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Scan for QR codes
    qr_codes = pyzbar.decode(img_array)
    
    # Extract QR code data
    qr_data = []
    for qr in qr_codes:
        if qr.type == 'QRCODE':
            try:
                data = qr.data.decode('utf-8')
                qr_data.append({
                    'data': data,
                    'type': 'QR Code',
                    'rect': (qr.rect.left, qr.rect.top, qr.rect.width, qr.rect.height)
                })
            except Exception as e:
                logger.error(f"Error decoding QR code: {e}")
    
    return qr_data
