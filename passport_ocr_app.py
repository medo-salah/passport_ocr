import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import io
import re
from pyzbar import pyzbar
from fpdf import FPDF
import easyocr
import torch
import logging
import subprocess
from spacy import util
import spacy

# ---- CONFIG ----
st.set_page_config(page_title="Passport OCR App", layout="centered")
st.title("📷 Passport OCR + Blur Detection + MRZ Extraction")
st.write("Upload a passport image to extract key info from MRZ and upper fields.")

# ---- ENHANCED GPU DETECTION ----
def get_gpu_info():
    """Returns detailed GPU information and troubleshooting tips"""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.version.cuda else "Not available",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "driver_issue": False,
        "troubleshooting_steps": []
    }
    
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        driver_version = result.stdout.strip() if result.returncode == 0 else "Unknown"
        gpu_info["nvidia_driver"] = driver_version
        
        if not gpu_info["cuda_available"]:
            gpu_info["driver_issue"] = True
            gpu_info["troubleshooting_steps"] = [
                "1. Verify CUDA Toolkit installation: Run `nvcc --version` in terminal",
                "2. Check PyTorch-CUDA compatibility: Your PyTorch requires CUDA " + gpu_info["cuda_version"],
                "3. Update NVIDIA drivers: Current driver - " + driver_version,
                "4. Reinstall PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`"
            ]
    except Exception:
        gpu_info["nvidia_driver"] = "Not detected"

    return gpu_info

# ---- GPU STATUS DISPLAY ----
gpu_info = get_gpu_info()
USE_GPU = gpu_info["cuda_available"]

if gpu_info["cuda_available"]:
    st.success(f"✅ CUDA GPU Detected: {gpu_info['gpu_name']} (Driver: {gpu_info['nvidia_driver']}, CUDA: {gpu_info['cuda_version']})")
else:
    if gpu_info["driver_issue"]:
        st.warning(f"🔧 NVIDIA Driver {gpu_info['nvidia_driver']} detected but PyTorch can't access GPU")
        with st.expander("Troubleshooting Guide"):
            st.write("PyTorch requires matching CUDA versions. Fix this by:")
            for step in gpu_info["troubleshooting_steps"]:
                st.write(step)
    else:
        st.warning("❌ No CUDA-capable GPU detected. Using CPU only")

# ---- LOGGING ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- LANGUAGE OPTIONS ----
lang = st.selectbox("Select OCR Language for General Text", ["English", "French", "Arabic"])
lang_map = {
    "English": "en",
    "French": "fr",
    "Arabic": "ar"
}

# ---- GPU TOGGLE ----
use_gpu = st.checkbox("Use GPU (if available)", value=USE_GPU)
if use_gpu and not USE_GPU:
    st.warning("⚠️ You selected GPU, but no CUDA-capable GPU is available. Falling back to CPU.")
    use_gpu = False

# ---- ENHANCED IMAGE PREPROCESSING ----
def preprocess_image(image, denoise=True, grayscale=True, enhance_contrast=True):
    """Apply preprocessing to improve OCR accuracy"""
    img_array = np.array(image)
    
    if grayscale:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    if enhance_contrast:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_array = clahe.apply(img_array) if grayscale else clahe.apply(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
    
    if denoise:
        img_array = cv2.fastNlMeansDenoising(img_array, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return Image.fromarray(img_array)

# ---- PREPROCESSING OPTIONS ----
st.sidebar.title("🛠️ Preprocessing Options")
do_rotate = st.sidebar.checkbox("Rotate Image 180°")
do_denoise = st.sidebar.checkbox("Denoise Image")
do_grayscale = st.sidebar.checkbox("Convert to Grayscale")
do_enhance = st.sidebar.checkbox("Enhance Contrast")

# ---- BLUR DETECTION ----
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm

# ---- MRZ EXTRACTION ----
def extract_mrz_fields(text):
    data = {
        "Document Type": "Not Found",
        "Issuing Country": "Not Found",
        "Full Name": "Not Found",
        "Date of Birth": "Not Found",
        "Sex": "Not Found",
        "Expiration Date": "Not Found",
        "Passport Number": "Not Found",
        "Nationality": "Not Found"
    }

    lines = [line.strip().replace(' ', '') for line in text.splitlines() if len(line.strip()) >= 30]
    if len(lines) >= 2:
        line1 = lines[-2]
        line2 = lines[-1]

        try:
            if line1.startswith("P<"):
                data["Document Type"] = line1[0]
                data["Issuing Country"] = line1[2:5]
                names = line1[5:].split("<<")
                surname = names[0].replace("<", " ").strip()
                given = names[1].replace("<", " ").strip() if len(names) > 1 else ""
                data["Full Name"] = f"{given} {surname}".strip()

            passport_number = re.sub(r'[^A-Z0-9]', '', line2[:9])
            nationality = line2[10:13]
            dob = line2[13:19]
            sex = line2[20]
            expiry = line2[21:27]

            dob_fmt = f"{dob[4:6]}-{dob[2:4]}-{dob[0:2]}" if len(dob) == 6 else dob
            expiry_fmt = f"{expiry[4:6]}-{expiry[2:4]}-{expiry[0:2]}" if len(expiry) == 6 else expiry

            data.update({
                "Passport Number": passport_number,
                "Nationality": nationality,
                "Date of Birth": dob_fmt,
                "Sex": sex,
                "Expiration Date": expiry_fmt
            })

        except Exception as e:
            logger.error(f"MRZ parsing error: {e}")

    return data

# ---- NER MODEL ----
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Model 'en_core_web_sm' not found. Run `python -m spacy download en_core_web_sm` to install.")
        st.stop()

nlp = load_spacy_model()

def extract_with_ner(text):
    doc = nlp(text)
    fields = {}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            fields["Name (NER)"] = ent.text
        elif ent.label_ == "GPE":
            fields.setdefault("Address (NER)", ent.text)
        elif ent.label_ == "DATE":
            fields.setdefault("Date (NER)", ent.text)
        elif ent.label_ == "NORP":
            fields.setdefault("Nationality (NER)", ent.text)
    return fields

# ---- BARCODE READER ----
def extract_barcodes(image):
    barcodes = pyzbar.decode(image)
    decoded = [barcode.data.decode("utf-8") for barcode in barcodes]
    return decoded

# ---- PDF EXPORT ----
def export_to_pdf(data, filename="passport_data.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Passport OCR Extracted Info", ln=True, align='C')
    pdf.ln(10)
    for key, value in data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()

# ---- MRZ DETECTION ----
def detect_mrz_region(image_cv):
    h, w = image_cv.shape[:2]
    bottom_half = image_cv[int(h * 0.5):h, :]
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_boxes = [cv2.boundingRect(c) for c in contours if 60 < cv2.boundingRect(c)[2] < w and 10 < cv2.boundingRect(c)[3] < 100]
    candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1])

    for (x, y, w_box, h_box) in candidate_boxes[::-1]:
        roi = gray[y:y+h_box, x:x+w_box]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            text_candidate = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
        except pytesseract.TesseractError:
            text_candidate = ""
        lines = [line for line in text_candidate.splitlines() if len(line.strip()) > 30]
        if len(lines) >= 2 and lines[0].startswith("P<"):
            return roi, text_candidate
    return gray[int(h*0.80):h, :], "⚠️ MRZ region not confidently detected"

# ---- MAIN APP ----
uploaded_file = st.file_uploader("Upload Passport Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    if do_rotate:
        image = image.rotate(180)
        logger.info("Rotated image 180 degrees")

    # Apply preprocessing
    image = preprocess_image(
        image,
        denoise=do_denoise,
        grayscale=do_grayscale,
        enhance_contrast=do_enhance
    )

    st.image(image, caption="Processed Passport Image", use_column_width=True)

    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if image_cv.shape[0] > 1200:
        scale = 1200 / image_cv.shape[0]
        image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)

    blurry, score = is_blurry(image)
    st.write(f"🔍 Blur Score: {score:.2f}")
    if blurry:
        st.error("❌ Image is too blurry! Please upload a clearer one.")
        st.stop()
    else:
        st.success("✅ Image is clear. Running OCR...")

    try:
        reader = easyocr.Reader([lang_map[lang]], gpu=use_gpu)
    except Exception as e:
        st.error(f"⚠️ EasyOCR failed to initialize with GPU. Falling back to CPU. Error: {e}")
        reader = easyocr.Reader([lang_map[lang]], gpu=False)

    results = reader.readtext(np.array(image), detail=1)
    text = "\n".join([r[1] for r in results])

    st.text_area("📄 Extracted Text (EasyOCR)", text if text.strip() else "⚠️ No text detected", height=200)

    mrz_image, mrz_text = detect_mrz_region(image_cv)
    st.text_area("MRZ OCR Text", mrz_text if mrz_text.strip() else "⚠️ No MRZ text detected", height=120)

    data = extract_mrz_fields(mrz_text)
    data.update(extract_with_ner(text))

    barcodes = extract_barcodes(image_cv)
    if barcodes:
        st.write("🔍 Detected Barcodes:")
        st.json(barcodes)
        data["Barcodes"] = ", ".join(barcodes)

    st.write("🧾 Extracted Info:")
    st.json(data)

    output_excel = io.BytesIO()
    df = pd.DataFrame([data])
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

    st.download_button("📅 Download Excel", output_excel.getvalue(), "passport_data.xlsx")

    pdf_bytes = export_to_pdf(data)
    st.download_button("📝 Download PDF", pdf_bytes, file_name="passport_data.pdf")
    
    
    
    
    
    
    
    
#streamlit run passport_ocr_app.py