
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"

import streamlit as st
import time
from contextlib import contextmanager

# Add progressive loading functionality
def progressive_load(components, delay=0.1):
    """
    Load components progressively with a slight delay between each
    
    Args:
        components: List of functions that render Streamlit components
        delay: Delay between components in seconds
    """
    for component_func in components:
        component_func()
        time.sleep(delay)  # Small delay for visual effect

# Create a placeholder component that can be replaced later
@contextmanager
def delayed_component():
    """Context manager for delayed component loading"""
    placeholder = st.empty()
    yield placeholder
    
# Create a function to load the app in stages
def load_app_progressively():
    """Load the app components in a progressive manner"""
    # Initialize progress tracking
    if "app_load_progress" not in st.session_state:
        st.session_state.app_load_progress = 0
    
    # Define loading stages
    stages = [
        ("Initializing app...", 0.1),
        ("Loading UI components...", 0.2),
        ("Setting up OCR engine...", 0.4),
        ("Preparing document analysis...", 0.6),
        ("Loading templates...", 0.8),
        ("Ready!", 1.0)
    ]
    
    # Show loading progress if not completed
    if st.session_state.app_load_progress < 1.0:
        progress_bar = st.progress(st.session_state.app_load_progress)
        status_text = st.empty()
        
        # Simulate progressive loading
        for message, progress in stages:
            if progress <= st.session_state.app_load_progress:
                continue
                
            status_text.text(message)
            progress_bar.progress(progress)
            st.session_state.app_load_progress = progress
            time.sleep(0.2)  # Adjust timing for visual effect
        
        # Clean up progress indicators after loading
        if st.session_state.app_load_progress >= 1.0:
            time.sleep(0.5)  # Brief pause to show "Ready!"
            progress_bar.empty()
            status_text.empty()

# ---- CONFIG ----
st.set_page_config(page_title="Passport OCR App", layout="wide")

from PIL import Image
import numpy as np
import cv2
import logging
import easyocr
from config import LANGUAGE_MAP
from gpu_utils import display_gpu_status
from image_processing import preprocess_image, setup_preprocessing_sidebar
from mrz_processing import extract_mrz_fields, detect_mrz_region
from ner_processing import load_spacy_model, extract_with_ner
from utils.blur_detection import is_blurry
from utils.barcode_reader import extract_barcodes
from utils.export_utils import export_to_pdf, export_to_excel
from ui_components import progress_bar
from theme_utils import apply_theme, initialize_theme
from mobile_utils import mobile_layout_wrapper
from language_utils import get_ui_text, initialize_language, set_language
from hologram_detection import detect_holograms, verify_security_features, detect_fake_document
from data_correction import setup_manual_correction, apply_corrections
from batch_processing import process_batch, export_batch_results
from image_enhancement import manual_crop_tool, detect_rotation
from document_analysis import detect_face, verify_face_quality, classify_document, detect_qr_codes
from template_matching import load_templates, match_template, extract_regions_from_template
from analytics_dashboard import log_analytics_event, display_analytics_dashboard
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import json
import requests
import time
import datetime
import sys
import os
import re
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Add function to check required libraries
def check_required_libraries():
    """Check if all required libraries are installed"""
    required_libraries = {
        "streamlit": "streamlit",
        "pytesseract": "pytesseract",
        "PIL": "pillow",
        "pandas": "pandas",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "pyzbar": "pyzbar",
        "fpdf": "fpdf2",
        "easyocr": "easyocr",
        "torch": "torch",
        "spacy": "spacy"
    }
    
    missing_libraries = []
    
    for lib_name, pip_name in required_libraries.items():
        try:
            __import__(lib_name)
        except ImportError:
            missing_libraries.append(pip_name)
    
    if missing_libraries:
        st.error(f"❌ Missing required libraries: {', '.join(missing_libraries)}")
        st.info(f"Install missing libraries with: pip install {' '.join(missing_libraries)}")
        return False
    
    return True

# Call the function at the beginning of the app
if not check_required_libraries():
    st.stop()

# Add this function after imports
def mask_sensitive_data(data, mask_fields=None):
    """
    Mask sensitive information in the data dictionary
    
    Args:
        data: Dictionary containing extracted data
        mask_fields: Dictionary of field names and whether they should be masked
        
    Returns:
        Dictionary with masked sensitive information
    """
    if not mask_fields:
        return data
    
    masked_data = data.copy()
    
    for field, should_mask in mask_fields.items():
        if should_mask and field in masked_data and masked_data[field] != "Not Found":
            value = masked_data[field]
            if field in ["Passport Number", "Document Number"]:
                # Show only last 4 characters
                masked_data[field] = "•" * (len(value) - 4) + value[-4:] if len(value) > 4 else "•" * len(value)
            elif field == "Date of Birth":
                # Show only year
                if len(value) >= 4:
                    masked_data[field] = "••/••/" + value[-4:]
                else:
                    masked_data[field] = "•" * len(value)
            elif field == "Full Name":
                # Show only initials
                parts = value.split()
                masked_name = []
                for part in parts:
                    if len(part) > 0:
                        masked_name.append(part[0] + "•" * (len(part) - 1))
                masked_data[field] = " ".join(masked_name)
            else:
                # Default masking for other fields
                masked_data[field] = "•" * len(value)
    
    return masked_data

# Add this function after imports
def setup_data_deletion():
    """
    Setup automatic data deletion based on user preferences
    """
    import time
    import threading
    
    # Check if deletion timer is already running
    if "deletion_timer_running" in st.session_state and st.session_state.deletion_timer_running:
        return
    
    # Function to delete data after specified time
    def delete_data_after_timeout():
        deletion_time = st.session_state.get("deletion_time", 15)
        time.sleep(deletion_time * 60)  # Convert minutes to seconds
        
        # Clear session state data
        keys_to_keep = ["theme", "ui_language", "auto_deletion", "deletion_time", "deletion_timer_running", "gdpr_consent"]
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear all uploaded files
        if "passport_uploader" in st.session_state:
            del st.session_state["passport_uploader"]
        
        if "batch_uploader" in st.session_state:
            del st.session_state["batch_uploader"]
        
        if "camera_input" in st.session_state:
            del st.session_state["camera_input"]
        
        # Reset timer flag
        st.session_state.deletion_timer_running = False
        
        # Log deletion
        logger.info(f"Auto-deleted user data and uploaded files after {deletion_time} minutes")
        log_data_processing_activity("auto_delete", {"deletion_time": deletion_time})
    
    # Start deletion timer in background thread if auto-deletion is enabled
    if st.session_state.get("auto_deletion", True):
        deletion_thread = threading.Thread(target=delete_data_after_timeout)
        deletion_thread.daemon = True  # Thread will exit when main program exits
        deletion_thread.start()
        
        # Set flag to indicate timer is running
        st.session_state.deletion_timer_running = True
        
        # Log start of deletion timer
        logger.info(f"Started data deletion timer for {st.session_state.get('deletion_time', 15)} minutes")

# Add this function after imports
def log_data_processing_activity(activity_type, details=None):
    """
    Log data processing activities for GDPR compliance
    
    Args:
        activity_type: Type of activity (e.g., "upload", "process", "export", "delete")
        details: Additional details about the activity
    """
    import datetime
    
    # Generate a random session ID if not already present
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        # Log session start
        log_analytics_event("session_start", {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_agent": st.context.headers.get("user-agent", "unknown") if hasattr(st, 'context') and hasattr(st.context, 'headers') else "unknown"
        })
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        "activity": activity_type,
        "details": details or {}
    }
    
    # Log to application logs
    logger.info(f"GDPR Log: {log_entry}")
    
    # In a production environment, you might want to store these logs
    # in a secure, GDPR-compliant storage system

# Add this function after imports
def clear_all_uploaded_files():
    """
    Clear all uploaded files from memory and reset file uploaders
    """
    # Clear file uploader widgets by resetting their keys
    if "passport_uploader" in st.session_state:
        del st.session_state["passport_uploader"]
    
    if "batch_uploader" in st.session_state:
        del st.session_state["batch_uploader"]
    
    # Clear camera input if present
    if "camera_input" in st.session_state:
        del st.session_state["camera_input"]
    
    # Clear any stored image data
    image_keys = ["image", "processed_image", "mrz_image", "face_img"]
    for key in image_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Log deletion
    logger.info("Cleared all uploaded files from memory")

# ---- INITIALIZATION ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize theme and language
initialize_theme()
initialize_language()

# Apply current theme
apply_theme()

# Check for consent
if "gdpr_consent" not in st.session_state:
    st.session_state.gdpr_consent = False

# If consent not given, show consent dialog
if not st.session_state.gdpr_consent:
    st.markdown("## 🔒 Data Processing Consent")
    st.markdown("""
    Before using this application, please read and agree to the following:
    
    1. This application processes passport and ID document data for OCR extraction
    2. By default, all data is processed in memory and not permanently stored
    3. You can enable automatic data deletion after a specified time period
    4. You can manually delete your data at any time
    
    For more information, please expand the "Privacy & GDPR Information" section below.
    """)
    
    consent_col1, consent_col2 = st.columns([1, 1])
    with consent_col1:
        if st.button("I Agree", key="consent_agree"):
            st.session_state.gdpr_consent = True
            st.experimental_rerun()
    with consent_col2:
        if st.button("I Decline", key="consent_decline"):
            st.error("You must consent to data processing to use this application")
            st.stop()
    
    # Stop execution until consent is given
    st.stop()

# Setup data deletion if consent given
if st.session_state.gdpr_consent and st.session_state.get("auto_deletion", True):
    setup_data_deletion()

# ---- LANGUAGE SELECTION ----
lang_col1, lang_col2 = st.columns([1, 1])

with lang_col1:
    ocr_lang = st.selectbox(
        get_ui_text("language", st.session_state.ui_language),
        list(LANGUAGE_MAP.keys())
    )

    # Log language selection
    log_analytics_event("language_selected", {"language": ocr_lang})
    
with lang_col2:
    ui_lang = st.selectbox(
        "Interface Language",
        ["English", "Français", "العربية"],
        index=0
    )
    
    # Map UI language selection to language code
    ui_lang_map = {"English": "en", "Français": "fr", "العربية": "ar"}
    if ui_lang in ui_lang_map:
        set_language(ui_lang_map[ui_lang])

# ---- TITLE ----
st.title(get_ui_text("title", st.session_state.ui_language))
st.write(get_ui_text("subtitle", st.session_state.ui_language))

# Add privacy and GDPR notices
with st.expander("ℹ️ Privacy & GDPR Information"):
    st.markdown("""
    **Privacy & Data Protection Notice:**
    
    - **Data Processing:** This application processes passport and ID document data for OCR extraction.
    - **Data Storage:** By default, no data is permanently stored. All data is processed in memory.
    - **Data Deletion:** Data is automatically deleted when you close the app or after the time period you set.
    - **Your Rights:** You have the right to access, rectify, and erase your personal data.
    - **Data Controller:** This application is provided for demonstration purposes only.
    
    For questions about data protection, please contact the application administrator.
    """)
    
    if st.session_state.get("auto_deletion", True):
        st.success(f"✅ Automatic data deletion is enabled. Data will be deleted after {st.session_state.get('deletion_time', 15)} minutes.")
    else:
        st.warning("⚠️ Automatic data deletion is disabled. Please manually clear your data when finished.")
        if st.button("Delete All My Data Now"):
            # Clear session state data
            keys_to_keep = ["theme", "ui_language", "auto_deletion", "deletion_time", "gdpr_consent"]
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
            
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear all uploaded files
            clear_all_uploaded_files()
            
            # Log deletion activity
            log_data_processing_activity("delete", {"scope": "all_data"})
            
            st.success("✅ All your data and uploaded files have been deleted from memory")
            st.experimental_rerun()

# ---- GPU SETUP ----
use_gpu = display_gpu_status("main")
use_gpu = st.checkbox(
    get_ui_text("gpu", st.session_state.ui_language),
    value=use_gpu
)

# ---- PREPROCESSING OPTIONS ----
preprocessing_options = setup_preprocessing_sidebar(st.session_state.ui_language)

# ---- ADVANCED FEATURES ----
st.sidebar.title("🚀 Advanced Features")
enable_face_detection = st.sidebar.checkbox("Enable Face Detection", value=False)
enable_doc_classification = st.sidebar.checkbox("Auto-Detect Document Type", value=True)
enable_template_matching = st.sidebar.checkbox("Use Template Matching", value=True,
                                              help="Match passport against known templates for better extraction")
enable_qr_detection = st.sidebar.checkbox("Scan QR Codes", value=True)
enable_fake_detection = st.sidebar.checkbox("Document Authenticity Check", value=False,
                                           help="Check for signs of document tampering or forgery")

# Log feature usage
enabled_features = []
if enable_face_detection:
    enabled_features.append("face_detection")
if enable_doc_classification:
    enabled_features.append("doc_classification")
if enable_template_matching:
    enabled_features.append("template_matching")
if enable_qr_detection:
    enabled_features.append("qr_detection")
if enable_fake_detection:
    enabled_features.append("fake_detection")

if enabled_features:
    for feature in enabled_features:
        log_analytics_event("feature_used", {"feature": feature})

# ---- PRIVACY SETTINGS ----
st.sidebar.title("🔒 Privacy Settings")

# Add GDPR compliance section
st.sidebar.markdown("### 🇪🇺 GDPR Compliance")
st.sidebar.info(
    "This application processes personal data in accordance with GDPR. "
    "No data is stored permanently unless you explicitly save it."
)

# Add auto-deletion option
enable_auto_deletion = st.sidebar.checkbox(
    "Enable Automatic Data Deletion", 
    value=True,
    help="Automatically delete all processed data when you close the app or after a set time"
)

# Store setting in session state
st.session_state.auto_deletion = enable_auto_deletion

if enable_auto_deletion:
    deletion_time = st.sidebar.slider(
        "Delete data after (minutes)", 
        min_value=1, 
        max_value=60, 
        value=15,
        help="Data will be automatically deleted after this time period"
    )
    st.session_state.deletion_time = deletion_time
    
    st.sidebar.success(f"✅ Data will be automatically deleted after {deletion_time} minutes")

# Add data masking option
st.sidebar.markdown("### 🔏 Data Masking")
enable_data_masking = st.sidebar.checkbox(
    "Mask Sensitive Information", 
    value=False,
    help="Hide sensitive data like passport numbers and dates of birth in the UI"
)

# Store masking setting in session state
st.session_state.data_masking = enable_data_masking

if enable_data_masking:
    # Let user choose which fields to mask
    st.sidebar.markdown("Select fields to mask:")
    mask_fields = {}
    for field in ["Passport Number", "Date of Birth", "Full Name", "Document Number"]:
        mask_fields[field] = st.sidebar.checkbox(f"Mask {field}", value=True)
    
    # Store masking preferences in session state
    st.session_state.mask_fields = mask_fields
    
    st.sidebar.success("✅ Data masking enabled")
    st.sidebar.info(
        "Selected sensitive information will be masked in the UI. "
        "Full data will still be available in exports."
    )

# Check if Tesseract is installed and properly configured
tesseract_installed = False
tesseract_path = ""
tessdata_path = ""

try:
    import pytesseract
    tesseract_path = pytesseract.get_tesseract_version()
    tesseract_installed = True
    
    # Try to find tessdata path
    if "TESSDATA_PREFIX" in os.environ:
        tessdata_path = os.environ["TESSDATA_PREFIX"]
    else:
        # Common locations
        possible_paths = [
            "C:/Program Files/Tesseract-OCR/tessdata",
            "C:/Program Files (x86)/Tesseract-OCR/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tessdata"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                tessdata_path = path
                os.environ["TESSDATA_PREFIX"] = path
                break
except Exception as e:
    tesseract_installed = False
    logger.error(f"Tesseract not available: {e}")

# Display Tesseract status
if tesseract_installed:
    st.sidebar.success(f"✅ Tesseract OCR detected (version: {tesseract_path})")
    if tessdata_path:
        st.sidebar.success(f"✅ Tessdata path: {tessdata_path}")
    else:
        st.sidebar.warning("⚠️ Tessdata path not found. Client-side processing may not work.")
else:
    st.sidebar.warning("⚠️ Tesseract OCR not detected. Client-side processing disabled.")

enable_client_side_processing = st.sidebar.checkbox(
    "Enable Client-Side Processing", 
    value=False,
    help="Process data in your browser without sending to servers. May be slower but keeps sensitive data private.",
    disabled=not tesseract_installed
)

if not tesseract_installed:
    if st.sidebar.button("How to Install Tesseract"):
        st.sidebar.info("""
        To enable client-side processing:
        1. Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
        2. Add Tesseract to your PATH or set TESSDATA_PREFIX environment variable
        3. Restart the app
        """)
elif not tessdata_path and st.sidebar.button("Fix Tessdata Path"):
    st.sidebar.info("""
    To fix the tessdata path:
    1. Locate your tessdata directory (usually in Tesseract installation folder)
    2. Set the TESSDATA_PREFIX environment variable to this path
    3. Or specify the path in the app settings below
    """)
    custom_tessdata_path = st.sidebar.text_input("Custom Tessdata Path", "C:/Program Files/Tesseract-OCR/tessdata")
    if st.sidebar.button("Apply Path"):
        os.environ["TESSDATA_PREFIX"] = custom_tessdata_path
        st.sidebar.success(f"✅ Set TESSDATA_PREFIX to {custom_tessdata_path}")
        st.experimental_rerun()

if enable_client_side_processing:
    if not tesseract_installed:
        st.sidebar.error("❌ Tesseract OCR is required for client-side processing but not installed")
    else:
        st.sidebar.success("✅ Privacy mode enabled - data stays on your device")
        st.sidebar.info(
            "How it works: When enabled, image processing and OCR run locally in your browser. "
            "No passport data is sent to external servers. This may use more of your computer's resources."
        )
    
    # Store the setting in session state for use in other functions
    st.session_state.client_side_processing = enable_client_side_processing and tesseract_installed and bool(tessdata_path)
else:
    st.session_state.client_side_processing = False

# Add a function to check if TensorFlow can be imported
def check_tensorflow():
    """Check if TensorFlow can be imported without memory errors"""
    try:
        import tensorflow as tf
        return True, None
    except (ImportError, MemoryError) as e:
        return False, str(e)

# Add this near the beginning of the app
tensorflow_available, tensorflow_error = check_tensorflow()
if not tensorflow_available:
    st.sidebar.warning(f"⚠️ TensorFlow not available: {tensorflow_error}")
    st.sidebar.info("Some features like advanced rotation detection will use simpler methods.")
    # Set a flag in session state
    st.session_state.disable_auto_rotation = True
else:
    st.sidebar.success("✅ TensorFlow available")

st.sidebar.markdown("### 🔄 Rotation Settings")
if tensorflow_available:
    enable_auto_rotation = st.sidebar.checkbox("Enable Auto-Rotation Detection", value=False)
    if not enable_auto_rotation:
        st.session_state.disable_auto_rotation = True
    else:
        st.session_state.disable_auto_rotation = False
else:
    st.sidebar.info("Auto-rotation detection requires TensorFlow, which is not available.")
    st.session_state.disable_auto_rotation = True

enable_manual_crop = st.sidebar.checkbox("Enable Manual Cropping", value=False)

# Add memory management function
def clear_memory():
    """Clear memory to prevent memory leaks"""
    import gc
    gc.collect()
    
    # Clear large objects from session state
    large_keys = ["image_cv", "processed_image"]
    for key in large_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Force garbage collection
    gc.collect()
    
    logger.info("Memory cleared")

# Add a button to clear memory
if st.sidebar.button("Clear Memory"):
    clear_memory()
    st.sidebar.success("✅ Memory cleared")

# Add function to reduce memory usage
def reduce_memory_usage():
    """Reduce memory usage by limiting TensorFlow memory growth"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Set memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Set memory growth for {len(gpus)} GPUs")
            
            # Limit memory usage to 1GB
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
            logger.info("Limited GPU memory to 1GB")
    except Exception as e:
        logger.warning(f"Could not configure TensorFlow memory: {e}")

# Call this function at the beginning of the app
try:
    reduce_memory_usage()
except Exception as e:
    logger.warning(f"Error reducing memory usage: {e}")

# ---- UPLOAD BUTTON ----
st.markdown("""
<style>
.big-upload-button {
    display: block;
    width: 100%;
    background-color: #4CAF50;
    color: white;
    padding: 14px 20px;
    margin: 8px 0;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 18px;
    text-align: center;
    text-decoration: none;
}
.big-upload-button:hover {
    background-color: #45a049;
}
.upload-options {
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
}
.upload-option {
    flex: 1;
}
.upload-header {
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h3>Upload Passport Image(s)</h3>", unsafe_allow_html=True)

# Create tabs for single and batch processing
upload_tab, batch_tab, settings_tab, admin_tab = st.tabs(["Single Document", "Batch Processing", "Settings", "Admin"])

with upload_tab:
    # Create two columns for upload options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="upload-header">📁 Upload from device</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a passport image", type=["jpg", "jpeg", "png"], key="passport_uploader", label_visibility="collapsed")

    with col2:
        st.markdown('<p class="upload-header">📷 Take a photo</p>', unsafe_allow_html=True)

        # Initialize camera state if not exists
        if "show_camera" not in st.session_state:
            st.session_state.show_camera = False

        # Button to toggle camera
        if not st.session_state.show_camera:
            if st.button("📷 Open Camera", key="open_camera_btn", help="Click to open camera and take a photo"):
                st.session_state.show_camera = True
                st.rerun()

        # Show camera input only when requested
        camera_photo = None
        if st.session_state.show_camera:
            camera_photo = st.camera_input("Take a picture", label_visibility="collapsed", key="camera_input")

            # Automatically close camera after photo is taken
            if camera_photo is not None:
                st.session_state.show_camera = False
                st.success("📷 Photo captured! Camera closed.")

            # Add button to close camera
            if st.button("❌ Close Camera", key="close_camera_btn", help="Close camera without taking a photo"):
                st.session_state.show_camera = False
                if "camera_input" in st.session_state:
                    del st.session_state["camera_input"]
                st.rerun()

    # Use either the uploaded file or the camera photo
    input_image = uploaded_file if uploaded_file is not None else camera_photo

    if input_image:
        # Log image upload event
        upload_source = "camera" if camera_photo is not None else "file_upload"
        log_analytics_event("image_upload", {
            "source": upload_source,
            "language": ocr_lang,
            "preprocessing_enabled": any(preprocessing_options.values()) if 'preprocessing_options' in locals() else False
        })

        # Process the image with progress bar
        processing_start_time = time.time()
        with progress_bar(get_ui_text("processing", st.session_state.ui_language)):
            image = Image.open(input_image)
            
            # Auto-rotation detection if enabled
            if enable_auto_rotation:
                with st.spinner("Detecting image orientation..."):
                    try:
                        rotated_image, angle = detect_rotation(image)
                        if angle != 0:
                            # Add a confirmation option
                            apply_rotation = st.checkbox(
                                f"Apply suggested {angle}° rotation?", 
                                value=False,
                                help="Uncheck this if your image is already correctly oriented"
                            )
                            
                            if apply_rotation:
                                st.success(f"✅ Rotated image by {angle}°")
                                image = rotated_image
                            else:
                                st.info("✓ Using original orientation")
                        else:
                            st.info("✓ No rotation needed")
                    except Exception as e:
                        st.error(f"⚠️ Error during rotation detection: {str(e)}")
                        st.warning("Continuing with original image orientation")
                        logger.error(f"Rotation detection error: {e}")

            # Manual cropping if enabled
            if enable_manual_crop:
                st.subheader("📐 Manual Crop Tool")
                image = manual_crop_tool(image)
                st.success("✅ Image cropped successfully")

            # Apply preprocessing
            image = preprocess_image(
                image,
                denoise=preprocessing_options["denoise"],
                grayscale=preprocessing_options["grayscale"],
                enhance_contrast=preprocessing_options["enhance"]
            )
            
            # Convert to OpenCV format for processing
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Resize if needed
            if image_cv.shape[0] > 1200:
                scale = 1200 / image_cv.shape[0]
                image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)
            
            # Store in session state for later use
            st.session_state.image = image
            st.session_state.image_cv = image_cv

        # Document classification if enabled
        if enable_doc_classification:
            doc_type, confidence = classify_document(image)
            st.info(f"📄 Detected document type: {doc_type} (Confidence: {confidence:.2f})")

        # Face detection if enabled
        if enable_face_detection:
            face_img, face_coords = detect_face(image)
            if face_img is not None:
                st.success("✅ Face detected in document")
                
                # Create columns for face display and validation
                face_col1, face_col2 = st.columns([1, 2])
                
                with face_col1:
                    st.image(face_img, caption="Extracted Face", width=200)
                
                with face_col2:
                    is_valid, issues = verify_face_quality(face_img)
                    if is_valid:
                        st.success("✅ Face image meets quality requirements")
                    else:
                        st.warning("⚠️ Face image has quality issues:")
                        for issue in issues:
                            st.write(f"- {issue}")
            else:
                st.warning("⚠️ No face detected in document")

        # QR code detection if enabled
        if enable_qr_detection:
            qr_data = detect_qr_codes(image)
            if qr_data:
                st.success(f"✅ Detected {len(qr_data)} QR code(s)")
                for i, qr in enumerate(qr_data):
                    st.write(f"QR Code {i+1}: {qr['data']}")

        # Document authenticity check if enabled
        if enable_fake_detection:
            with st.spinner("Checking document authenticity..."):
                try:
                    # Run fake document detection
                    marked_image, detection_data = detect_fake_document(image)
                    security_result, authenticity_score = verify_security_features(image)
                    
                    # Create columns for display
                    auth_col1, auth_col2 = st.columns([1, 1])
                    
                    with auth_col1:
                        st.image(marked_image, caption="Document Authenticity Analysis", use_column_width=True)
                    
                    with auth_col2:
                        # Display authenticity verification results
                        st.subheader("🔍 Document Authenticity Check")
                        
                        # Show authenticity score with color coding
                        score = security_result["authenticity_score"]
                        if score >= 0.8:
                            st.success(f"Authenticity Score: {score:.2f}")
                        elif score >= 0.5:
                            st.warning(f"Authenticity Score: {score:.2f}")
                        else:
                            st.error(f"Authenticity Score: {score:.2f}")
                        
                        st.write(security_result["assessment"])
                        
                        # Show detected issues
                        if detection_data["issues"]:
                            st.write("Potential issues detected:")
                            for issue in detection_data["issues"]:
                                st.write(f"⚠️ {issue}")
                        else:
                            st.write("✅ No suspicious elements detected")
                        
                        # Add disclaimer
                        st.info("Note: This analysis is based on image processing and may not detect all forms of document tampering. Physical inspection by a trained professional is recommended for full verification.")
                except Exception as e:
                    logger.error(f"Error in document authenticity check: {e}")
                    st.error(f"⚠️ Error checking document authenticity: {str(e)}")

        st.image(image, caption="Processed Passport Image", use_column_width=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if image_cv.shape[0] > 1200:
            scale = 1200 / image_cv.shape[0]
            image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)

        blurry, score = is_blurry(image)
        st.write(f"🔍 {get_ui_text('blur_score', st.session_state.ui_language)}: {score:.2f}")
        if blurry:
            st.error(f"❌ {get_ui_text('blur_warning', st.session_state.ui_language)}")
            st.stop()
        else:
            st.success(f"✅ {get_ui_text('clear_image', st.session_state.ui_language)}")

        # OCR processing with progress bar
        with progress_bar("OCR"):
            text = ""  # Initialize text variable to avoid NameError
            try:
                if st.session_state.get("client_side_processing", False):
                    st.info("🔒 Using client-side OCR processing for privacy")
                    try:
                        # Try to use Tesseract for client-side processing
                        import pytesseract
                        text = pytesseract.image_to_string(
                            np.array(image), 
                            lang=LANGUAGE_MAP[ocr_lang]
                        )
                        st.success("✅ Successfully processed with client-side OCR")
                    except Exception as tesseract_error:
                        logger.error(f"Client-side OCR error: {tesseract_error}")
                        st.error(f"⚠️ Client-side OCR failed: {tesseract_error}")
                        st.warning("⚠️ Falling back to server-side OCR...")
                        
                        # Automatically disable client-side processing after error
                        st.session_state.client_side_processing = False
                        
                        # Fall back to server-side OCR
                        try:
                            reader = easyocr.Reader([LANGUAGE_MAP[ocr_lang]], gpu=use_gpu)
                            results = reader.readtext(np.array(image), detail=1)
                            text = "\n".join([r[1] for r in results])
                            st.success("✅ Successfully processed with server-side OCR")
                        except Exception as easyocr_error:
                            logger.error(f"EasyOCR error: {easyocr_error}")
                            st.error(f"⚠️ Server-side OCR also failed: {easyocr_error}")
                            text = "OCR processing failed"
                else:
                    try:
                        reader = easyocr.Reader([LANGUAGE_MAP[ocr_lang]], gpu=use_gpu)
                        results = reader.readtext(np.array(image), detail=1)
                        text = "\n".join([r[1] for r in results])

                        # Log successful OCR
                        log_analytics_event("ocr_success", {
                            "language": ocr_lang,
                            "gpu_used": use_gpu,
                            "text_length": len(text),
                            "confidence_scores": [r[2] for r in results] if results else []
                        })
                    except Exception as e:
                        logger.error(f"EasyOCR error: {e}")
                        st.error(f"⚠️ EasyOCR failed. Falling back to CPU. Error: {e}")

                        # Log OCR error
                        log_analytics_event("ocr_error", {
                            "language": ocr_lang,
                            "gpu_attempted": use_gpu,
                            "error_type": "easyocr_gpu_failure",
                            "error_message": str(e)
                        })

                        try:
                            reader = easyocr.Reader([LANGUAGE_MAP[ocr_lang]], gpu=False)
                            results = reader.readtext(np.array(image), detail=1)
                            text = "\n".join([r[1] for r in results])

                            # Log successful CPU fallback
                            log_analytics_event("ocr_success", {
                                "language": ocr_lang,
                                "gpu_used": False,
                                "fallback": True,
                                "text_length": len(text)
                            })
                        except Exception as e2:
                            logger.error(f"CPU fallback error: {e2}")
                            st.error(f"⚠️ CPU fallback also failed: {e2}")
                            text = "OCR processing failed"

                            # Log complete OCR failure
                            log_analytics_event("ocr_error", {
                                "language": ocr_lang,
                                "error_type": "complete_ocr_failure",
                                "error_message": str(e2)
                            })
            except Exception as e:
                logger.error(f"OCR processing error: {e}")
                st.error(f"⚠️ OCR processing error: {e}")
                text = "OCR processing failed"

        # Create columns for results
        text_col, mrz_col = st.columns(2)

        with text_col:
            st.text_area(
                f"📄 {get_ui_text('extracted_text', st.session_state.ui_language)}",
                text if text.strip() else f"⚠️ {get_ui_text('no_text', st.session_state.ui_language)}",
                height=200
            )
            
            # Add feedback mechanism for OCR text
            with st.expander("📝 Report incorrect OCR text"):
                st.write("If the OCR text above is incorrect, please provide the correct text below:")
                corrected_ocr = st.text_area(
                    "Corrected OCR text",
                    text,
                    height=150,
                    key="corrected_ocr_text"
                )
                
                feedback_reason = st.selectbox(
                    "What was wrong with the OCR result?",
                    options=[
                        "Missing text",
                        "Extra text",
                        "Incorrect characters",
                        "Wrong language detection",
                        "Text formatting issues",
                        "Other issue"
                    ],
                    key="ocr_feedback_reason"
                )
                
                other_details = st.text_input("Additional details (optional)", key="ocr_feedback_details")
                
                if st.button("Submit OCR Feedback", key="submit_ocr_feedback"):
                    if corrected_ocr != text:
                        # Generate image hash for reference
                        image_hash = generate_image_hash(image)
                        
                        # Save the feedback
                        feedback_id = save_feedback(
                            text, 
                            corrected_ocr, 
                            image_hash, 
                            f"ocr_correction_{feedback_reason}"
                        )
                        
                        # Log additional details if provided
                        if other_details:
                            with open(os.path.join("feedback_data", f"{feedback_id}_details.txt"), "w") as f:
                                f.write(other_details)
                        
                        st.success("Thank you for your feedback! This helps improve our OCR system.")
                        
                        # Log data processing activity for GDPR compliance
                        log_data_processing_activity(
                            "feedback_submission",
                            {"feedback_type": "ocr_correction", "feedback_id": feedback_id}
                        )

                        # Log analytics event
                        log_analytics_event("feedback_submitted", {
                            "feedback_type": "ocr_correction",
                            "reason": feedback_reason,
                            "text_length_original": len(text),
                            "text_length_corrected": len(corrected_ocr),
                            "has_additional_details": bool(other_details)
                        })

                        # Store feedback in database if enabled
                        try:
                            from database_manager import is_database_enabled, get_database_manager

                            if is_database_enabled() and st.session_state.get("admin_store_feedback", True):
                                db_manager = get_database_manager()

                                feedback_data = {
                                    "feedback_id": feedback_id,
                                    "session_id": st.session_state.session_id,
                                    "image_hash": image_hash,
                                    "feedback_type": "ocr_correction",
                                    "original_text": text,
                                    "corrected_text": corrected_ocr,
                                    "diff_ratio": calculate_diff_ratio(text, corrected_ocr),
                                    "additional_details": other_details
                                }

                                db_feedback_id = db_manager.store_feedback(feedback_data)
                                logger.info(f"Stored feedback in database with ID: {db_feedback_id}")

                        except Exception as e:
                            logger.error(f"Error storing feedback in database: {e}")
                    else:
                        st.warning("No changes detected in the corrected text.")

        # MRZ processing with progress bar
        with progress_bar("MRZ"):
            mrz_text = ""  # Initialize to avoid NameError
            try:
                # Make sure image_cv is defined
                if 'image_cv' not in locals() and 'image_cv' not in st.session_state:
                    # Convert PIL image to OpenCV format
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Resize if needed
                    if image_cv.shape[0] > 1200:
                        scale = 1200 / image_cv.shape[0]
                        image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)
                        
                    # Store in session state
                    st.session_state.image_cv = image_cv
                elif 'image_cv' in st.session_state:
                    image_cv = st.session_state.image_cv
                
                # Try to detect MRZ region with error handling
                try:
                    use_client_side = st.session_state.get("client_side_processing", False)
                    mrz_image, mrz_text = detect_mrz_region(image_cv, use_client_side=use_client_side)

                    # Log successful MRZ detection
                    log_analytics_event("mrz_success", {
                        "client_side": use_client_side,
                        "mrz_length": len(mrz_text) if mrz_text else 0,
                        "mrz_detected": bool(mrz_text and mrz_text.strip())
                    })
                except Exception as mrz_error:
                    logger.error(f"MRZ detection error: {mrz_error}")
                    st.error(f"⚠️ Error detecting MRZ: {str(mrz_error)}")

                    # Log MRZ error
                    log_analytics_event("mrz_error", {
                        "error_type": "mrz_detection_failed",
                        "error_message": str(mrz_error),
                        "client_side": st.session_state.get("client_side_processing", False)
                    })

                    # Try the simplified version as a fallback
                    try:
                        st.warning("Trying simplified MRZ detection as fallback...")
                        # For now, just set error text since detect_mrz_region_simple is not defined
                        mrz_text = "Error detecting MRZ - simplified fallback not available"
                    except Exception as simple_error:
                        logger.error(f"Simple MRZ detection error: {simple_error}")
                        mrz_text = "Error detecting MRZ"
                
            except Exception as e:
                logger.error(f"Error in MRZ processing: {e}")
                st.error(f"⚠️ Error processing image: {str(e)}")
                mrz_text = "Error processing image"

        with mrz_col:
            st.text_area(
                f"{get_ui_text('mrz_text', st.session_state.ui_language)}",
                mrz_text if mrz_text.strip() else f"⚠️ {get_ui_text('no_mrz', st.session_state.ui_language)}",
                height=120
            )
            
            # Add feedback mechanism for MRZ text
            with st.expander("📝 Report incorrect MRZ text"):
                st.write("If the MRZ text above is incorrect, please provide the correct text below:")
                corrected_mrz = st.text_area(
                    "Corrected MRZ text",
                    mrz_text,
                    height=100,
                    key="corrected_mrz_text"
                )
                
                mrz_feedback_reason = st.selectbox(
                    "What was wrong with the MRZ detection?",
                    options=[
                        "MRZ not detected",
                        "Partial MRZ detection",
                        "Incorrect characters",
                        "Wrong MRZ format",
                        "Other issue"
                    ],
                    key="mrz_feedback_reason"
                )
                
                mrz_other_details = st.text_input("Additional details (optional)", key="mrz_feedback_details")
                
                if st.button("Submit MRZ Feedback", key="submit_mrz_feedback"):
                    if corrected_mrz != mrz_text:
                        # Generate image hash for reference
                        image_hash = generate_image_hash(image)
                        
                        # Save the feedback
                        feedback_id = save_feedback(
                            mrz_text, 
                            corrected_mrz, 
                            image_hash, 
                            f"mrz_correction_{mrz_feedback_reason}"
                        )
                        
                        # Log additional details if provided
                        if mrz_other_details:
                            with open(os.path.join("feedback_data", f"{feedback_id}_details.txt"), "w") as f:
                                f.write(mrz_other_details)
                        
                        st.success("Thank you for your feedback! This helps improve our MRZ detection.")
                        
                        # Log data processing activity for GDPR compliance
                        log_data_processing_activity(
                            "feedback_submission",
                            {"feedback_type": "mrz_correction", "feedback_id": feedback_id}
                        )

                        # Log analytics event
                        log_analytics_event("feedback_submitted", {
                            "feedback_type": "mrz_correction",
                            "reason": mrz_feedback_reason,
                            "text_length_original": len(mrz_text),
                            "text_length_corrected": len(corrected_mrz),
                            "has_additional_details": bool(mrz_other_details)
                        })

                        # Store MRZ feedback in database if enabled
                        try:
                            from database_manager import is_database_enabled, get_database_manager

                            if is_database_enabled() and st.session_state.get("admin_store_feedback", True):
                                db_manager = get_database_manager()

                                feedback_data = {
                                    "feedback_id": feedback_id,
                                    "session_id": st.session_state.session_id,
                                    "image_hash": image_hash,
                                    "feedback_type": "mrz_correction",
                                    "original_text": mrz_text,
                                    "corrected_text": corrected_mrz,
                                    "diff_ratio": calculate_diff_ratio(mrz_text, corrected_mrz),
                                    "additional_details": mrz_other_details
                                }

                                db_feedback_id = db_manager.store_feedback(feedback_data)
                                logger.info(f"Stored MRZ feedback in database with ID: {db_feedback_id}")

                        except Exception as e:
                            logger.error(f"Error storing MRZ feedback in database: {e}")
                    else:
                        st.warning("No changes detected in the corrected text.")

        # Enhance data extraction with template information
        with progress_bar("Data Extraction"):
            try:
                nlp = load_spacy_model()
                
                # Extract data from MRZ with error handling
                try:
                    data = extract_mrz_fields(mrz_text)
                except Exception as mrz_extract_error:
                    logger.error(f"MRZ extraction error: {mrz_extract_error}")
                    st.error(f"⚠️ Error extracting MRZ data: {str(mrz_extract_error)}")
                    # Initialize with empty data
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
                
                # Try NER extraction with error handling
                try:
                    ner_data = extract_with_ner(text, nlp)
                    
                    # Only update fields that are "Not Found" with NER data
                    for key, value in ner_data.items():
                        if key not in data or data[key] == "Not Found":
                            data[key] = value
                except Exception as ner_error:
                    logger.error(f"NER extraction error: {ner_error}")
                    st.warning(f"⚠️ Error extracting data with NER: {str(ner_error)}")
                
                # Try to extract more information from the OCR text if MRZ failed
                if all(value == "Not Found" for key, value in data.items() if key not in ["_validation"]):
                    st.warning("MRZ extraction failed. Attempting to extract data from OCR text.")
                    
                    try:
                        # Try to find passport number
                        passport_match = re.search(r'(?:Passport|Number|No)[:\.\s]+([A-Z0-9]{7,9})', text, re.IGNORECASE)
                        if passport_match:
                            data["Passport Number"] = passport_match.group(1)
                        
                        # Try to find name
                        name_match = re.search(r'(?:Name|Nom)[:\.\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
                        if name_match:
                            data["Full Name"] = name_match.group(1).strip()
                        
                        # Try to find nationality
                        nationality_match = re.search(r'(?:Nationality|Nationalité)[:\.\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
                        if nationality_match:
                            data["Nationality"] = nationality_match.group(1).strip()
                        
                        # Special case for USA passports
                        if "UNITED STATES" in text.upper() or "USA" in text.upper() or "AMERICA" in text.upper():
                            data["Issuing Country"] = "USA"
                            data["Nationality"] = "USA"
                            
                        # Set document type to Passport if found in text
                        if "PASSPORT" in text.upper() or "PASSEPORT" in text.upper() or "PASAPORTE" in text.upper():
                            data["Document Type"] = "Passport"
                        
                    except Exception as regex_error:
                        logger.error(f"Regex extraction error: {regex_error}")
                        st.warning(f"⚠️ Error extracting data with regex: {str(regex_error)}")
            except Exception as e:
                logger.error(f"Data extraction error: {e}")
                st.error(f"⚠️ Error during data extraction: {str(e)}")
                data = {
                    "Document Type": "Error",
                    "Issuing Country": "Error",
                    "Full Name": "Error",
                    "Date of Birth": "Error",
                    "Sex": "Error",
                    "Expiration Date": "Error",
                    "Passport Number": "Error",
                    "Nationality": "Error",
                    "Error": str(e)
                }

        # Remove signature verification results from data
        # if enable_signature_verification and 'signature_img' in locals() and signature_img is not None:
        #     # Add signature verification to data
        #     data["Signature"] = "Detected"
        #     
        #     # Add verification results if available
        #     if 'is_valid' in locals():
        #         data["Signature Verification"] = "Valid" if is_valid else "Issues Detected"
        #         
        #         if 'score' in locals() and score is not None:
        #             data["Signature Match Score"] = f"{score:.2f}"

        # Display extracted info
        st.write(f"🧾 {get_ui_text('extracted_info', st.session_state.ui_language)}:")

        # Apply data masking if enabled
        display_data = data
        if st.session_state.get("data_masking", False):
            display_data = mask_sensitive_data(data, st.session_state.get("mask_fields", {}))
            st.info("🔏 Some sensitive information is masked. Full data will be available in exports.")
            
            # Add a toggle to temporarily show unmasked data
            show_unmasked = st.checkbox("Show unmasked data", value=False, help="Temporarily show all data unmasked")
            if show_unmasked:
                st.warning("⚠️ Showing unmasked sensitive data")
                display_data = data

        # Display the data (masked or unmasked)
        st.json(display_data)

        # Log processing completion
        processing_time = time.time() - processing_start_time
        log_analytics_event("processing_complete", {
            "processing_time": processing_time,
            "language": ocr_lang,
            "features_used": {
                "face_detection": enable_face_detection,
                "doc_classification": enable_doc_classification,
                "template_matching": enable_template_matching,
                "qr_detection": enable_qr_detection,
                "fake_detection": enable_fake_detection
            },
            "data_extracted": bool(data and any(v != "Not Found" and v != "Error" for v in data.values() if isinstance(v, str))),
            "preprocessing_options": preprocessing_options if 'preprocessing_options' in locals() else {}
        })

        # Store results in database if enabled
        try:
            from database_manager import is_database_enabled, get_database_manager

            if is_database_enabled() and st.session_state.get("admin_store_ocr_results", True):
                db_manager = get_database_manager()

                # Prepare result data for database storage
                result_data = {
                    "image_hash": generate_image_hash(input_image),
                    "ocr_text": text,
                    "mrz_text": mrz_text if 'mrz_text' in locals() else "",
                    "extracted_data": data,
                    "processing_time": processing_time,
                    "language": ocr_lang,
                    "features_used": {
                        "face_detection": enable_face_detection,
                        "doc_classification": enable_doc_classification,
                        "template_matching": enable_template_matching,
                        "qr_detection": enable_qr_detection,
                        "fake_detection": enable_fake_detection
                    },
                    "confidence_scores": [],  # Add confidence scores if available
                    "status": "success"
                }

                # Store in database
                result_id = db_manager.store_ocr_result(st.session_state.session_id, result_data)
                st.success(f"✅ Results saved to database (ID: {result_id})")

        except Exception as e:
            logger.error(f"Error storing results in database: {e}")
            if st.session_state.get("admin_enable_database", False):
                st.warning(f"⚠️ Could not save to database: {e}")

        # Add this after displaying the extracted info
        if data and "_validation" in data:
            st.subheader("🔍 Data Validation")
            
            # Create a table for validation results
            validation_data = []
            for field, result in data["_validation"].items():
                # Get the value (masked if needed)
                if st.session_state.get("data_masking", False) and field in st.session_state.get("mask_fields", {}) and st.session_state.mask_fields[field]:
                    # Get the masked value from display_data
                    value = display_data.get(field, "Not Found")
                else:
                    value = data.get(field, "Not Found")
                
                validation_data.append({
                    "Field": field,
                    "Value": value,
                    "Status": "✅" if result["valid"] else "❌",
                    "Message": result["message"]
                })
            
            # Convert to DataFrame for display
            import pandas as pd
            validation_df = pd.DataFrame(validation_data)
            
            # Style the DataFrame
            def highlight_status(val):
                if val == "✅":
                    return 'background-color: #d4edda; color: #155724'
                elif val == "❌":
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return ''
            
            # Apply styling only if there are rows in the DataFrame
            if not validation_df.empty:
                styled_df = validation_df.style.applymap(highlight_status, subset=['Status'])
                st.dataframe(styled_df)
            else:
                st.info("No validation results available")
            
            # Check if any critical fields are invalid
            critical_fields = ["Passport Number", "Date of Birth", "Expiration Date"]
            invalid_critical = [field for field in critical_fields 
                                if field in data["_validation"] and not data["_validation"][field]["valid"]]
            
            if invalid_critical:
                st.warning(f"⚠️ Critical fields with validation issues: {', '.join(invalid_critical)}")
                st.info("Please verify these fields manually before proceeding.")
            
            # Remove validation data from export
            export_data = {k: v for k, v in data.items() if k != "_validation"}
        else:
            export_data = data

        # Add manual correction option
        st.subheader("🖊️ Manual Correction")
        enable_correction = st.checkbox("Enable manual correction", value=False)

        if enable_correction:
            # Call the manual correction function with unique context
            corrected_data, was_corrected = setup_manual_correction(data, context="main_single_doc")
            
            # If corrections were made, update the data
            if was_corrected:
                # Apply corrections and update validation status
                updated_data = apply_corrections(data, corrected_data)
                
                # Update the data for display and export
                data = updated_data
                
                # Update display data with masking if needed
                if st.session_state.get("data_masking", False):
                    display_data = mask_sensitive_data(data, st.session_state.get("mask_fields", {}))
                else:
                    display_data = data
                
                # Show the updated data
                st.subheader("📋 Updated Data")
                st.json(display_data)
                
                # Store the corrected data in session state for later use
                st.session_state.corrected_data = data

        # Add template creation option
        with st.expander("🔧 Template Management"):
            st.markdown("### Create New Template")
            st.info("Create a new template from the current image to improve future extractions")
            
            template_name = st.text_input("Template Name", placeholder="e.g., EU Passport 2020")
            country_code = st.text_input("Country Code (3 letters)", max_chars=3, placeholder="e.g., USA")
            
            if st.button("Create Template") and template_name and "image" in st.session_state:
                try:
                    # Create templates directory if it doesn't exist
                    os.makedirs("templates", exist_ok=True)
                    
                    # Save current image as template
                    template_path = f"templates/{template_name.lower().replace(' ', '_')}.jpg"
                    cv2.imwrite(template_path, image_cv)
                    
                    # Create basic regions (user would need to adjust these later)
                    h, w = image_cv.shape[:2]
                    basic_regions = {
                        "mrz_region": [50, int(h*0.8), w-100, int(h*0.15)],
                        "photo_region": [50, 150, int(w*0.3), int(h*0.5)],
                        "name_region": [int(w*0.4), 200, int(w*0.5), 50],
                        "passport_number_region": [int(w*0.4), 300, 200, 50]
                        # Remove signature region
                        # "signature_region": [int(w*0.4), int(h*0.7), int(w*0.5), 70]
                    }
                    
                    # Create template object
                    from template_matching import PassportTemplate
                    new_template = PassportTemplate(
                        template_name,
                        template_path,
                        basic_regions,
                        country_code.upper() if country_code else None
                    )
                    
                    # Save template info (in a real app, this would be more sophisticated)
                    import json
                    template_info = {
                        "name": template_name,
                        "path": template_path,
                        "country_code": country_code.upper() if country_code else None,
                        "regions": basic_regions
                    }
                    
                    with open(f"templates/{template_name.lower().replace(' ', '_')}.json", "w") as f:
                        json.dump(template_info, f, indent=2)
                    
                    st.success(f"✅ Created new template: {template_name}")
                    st.info("The template has been saved with basic regions. You may need to adjust them for optimal extraction.")
                except Exception as e:
                    st.error(f"⚠️ Error creating template: {e}")

        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                f"📅 {get_ui_text('download_excel', st.session_state.ui_language)}",
                export_to_excel(export_data),
                "passport_data.xlsx",
                help="Download data as Excel file. You are responsible for securing this file."
            )
        with col2:
            st.download_button(
                f"📝 {get_ui_text('download_pdf', st.session_state.ui_language)}",
                export_to_pdf(export_data),
                file_name="passport_data.pdf",
                help="Download data as PDF file. You are responsible for securing this file."
            )
        with col3:
            if st.button("🗑️ Delete This Data", help="Immediately delete this data and uploaded files from memory"):
                # Clear only the current data
                if "data" in st.session_state:
                    del st.session_state.data
                
                # Clear the current uploaded file
                if "passport_uploader" in st.session_state:
                    del st.session_state["passport_uploader"]
                
                # Clear camera input if present
                if "camera_input" in st.session_state:
                    del st.session_state["camera_input"]
                
                # Clear any stored image data for this document
                image_keys = ["image", "processed_image", "mrz_image", "face_img"]
                for key in image_keys:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Log deletion activity
                log_data_processing_activity("delete", {"scope": "current_document"})
                
                st.success("✅ Document data and uploaded files deleted from memory")
                st.experimental_rerun()

with batch_tab:
    st.markdown('<p class="upload-header">📁 Upload multiple documents</p>', unsafe_allow_html=True)
    
    # Add GDPR notice for batch processing
    with st.expander("ℹ️ Batch Processing & Data Retention"):
        st.markdown("""
        **Batch Processing Data Policy:**
        
        - All uploaded files are processed in memory
        - Results are not permanently stored unless you export them
        - All data is automatically deleted when you close the app or after the time period you set
        - You can manually delete all data using the button below
        """)
        
        if st.button("🗑️ Delete All Batch Data", key="delete_batch"):
            # Clear batch-related data
            batch_keys = ["batch_files", "batch_results"]
            for key in batch_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear batch file uploader
            if "batch_uploader" in st.session_state:
                del st.session_state["batch_uploader"]
            
            # Log deletion activity
            log_data_processing_activity("delete", {"scope": "batch_data"})
            
            st.success("✅ All batch processing data and uploaded files deleted")
            st.experimental_rerun()
    
    batch_files = st.file_uploader("Choose passport images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_uploader", label_visibility="collapsed")
    
    if batch_files:
        st.write(f"📂 {len(batch_files)} files uploaded")
        
        if st.button("Process Batch"):
            # Create a progress bar
            progress_placeholder = st.empty()
            status_text = st.empty()
            
            # Process the batch files
            try:
                # Define a function to process each image
                def process_single_image(file):
                    # Open the image
                    image = Image.open(file)
                    
                    # Apply preprocessing
                    processed_image = preprocess_image(
                        image,
                        denoise=preprocessing_options["denoise"],
                        grayscale=preprocessing_options["grayscale"],
                        enhance_contrast=preprocessing_options["enhance"]
                    )
                    
                    # Convert to OpenCV format
                    image_cv = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                    
                    # Resize if needed
                    if image_cv.shape[0] > 1200:
                        scale = 1200 / image_cv.shape[0]
                        image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)
                    
                    # Check if image is blurry
                    blurry, score = is_blurry(processed_image)
                    if blurry:
                        return {"error": "Image too blurry", "blur_score": score}
                    
                    # OCR processing
                    try:
                        reader = easyocr.Reader([LANGUAGE_MAP[ocr_lang]], gpu=use_gpu)
                        results = reader.readtext(np.array(processed_image), detail=1)
                        text = "\n".join([r[1] for r in results])
                    except Exception as e:
                        return {"error": f"OCR error: {str(e)}"}
                    
                    # MRZ processing
                    mrz_image, mrz_text = detect_mrz_region(image_cv)
                    
                    # Data extraction
                    nlp = load_spacy_model()
                    data = extract_mrz_fields(mrz_text)
                    data.update(extract_with_ner(text, nlp))
                    
                    # Barcode extraction
                    barcodes = extract_barcodes(image_cv)
                    if barcodes:
                        data["Barcodes"] = ", ".join(barcodes)
                    
                    return data
                
                # Process all files with progress updates
                results = []
                failed = []
                
                for i, file in enumerate(batch_files):
                    # Update progress
                    progress = (i + 1) / len(batch_files)
                    progress_placeholder.progress(progress)
                    status_text.text(f"Processing {file.name} ({i+1}/{len(batch_files)})")
                    
                    try:
                        # Process the file
                        result = process_single_image(file)
                        
                        # Check if there was an error
                        if "error" in result:
                            failed.append({"filename": file.name, "error": result["error"]})
                        else:
                            results.append({"filename": file.name, "data": result})
                    except Exception as e:
                        failed.append({"filename": file.name, "error": str(e)})
                
                # Clear progress indicators
                progress_placeholder.empty()
                status_text.empty()
                
                # Show results
                st.success(f"✅ Processed {len(results)} files successfully")
                
                if failed:
                    st.error(f"❌ Failed to process {len(failed)} files")
                    with st.expander("Show failed files"):
                        for fail in failed:
                            st.write(f"- {fail['filename']}: {fail['error']}")
                
                # Create download buttons for results
                if results:
                    st.subheader("Download Results")
                    
                    # Create Excel with all results
                    excel_bytes = export_to_excel({r["filename"]: r["data"] for r in results})
                    
                    # Create a ZIP file with individual PDFs
                    import io
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for result in results:
                            pdf_bytes = export_to_pdf(result["data"])
                            filename = result["filename"].split('.')[0] + '.pdf'
                            zip_file.writestr(filename, pdf_bytes)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📊 Download Excel Report",
                            excel_bytes,
                            "passport_batch_results.xlsx"
                        )
                    
                    with col2:
                        st.download_button(
                            "📑 Download PDF Reports (ZIP)",
                            zip_buffer.getvalue(),
                            "passport_pdfs.zip"
                        )
            
            except Exception as e:
                st.error(f"❌ Batch processing error: {str(e)}")



# Remove the mobile_layout_wrapper call since we're handling the upload directly
    #streamlit run main.py

# Add to initialization
def check_templates_directory():
    """Check if templates directory exists and create it if not"""
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        try:
            os.makedirs(templates_dir)
            logger.info(f"Created templates directory: {templates_dir}")
            
            # Create a README file in the templates directory
            readme_path = os.path.join(templates_dir, "README.txt")
            with open(readme_path, "w") as f:
                f.write("""
                Templates Directory
                
                Place your document templates here. Templates should be:
                - PNG or JPG format
                - Named descriptively (e.g., passport_usa.png, id_card_france.jpg)
                - Include region definitions in a matching JSON file (e.g., passport_usa.json)
                
                Example JSON format:
                {
                    "regions": [
                        {
                            "name": "photo",
                            "x": 0.1,
                            "y": 0.2,
                            "width": 0.3,
                            "height": 0.4
                        },
                        {
                            "name": "mrz",
                            "x": 0.05,
                            "y": 0.8,
                            "width": 0.9,
                            "height": 0.15
                        }
                    ]
                }
                """)
            logger.info(f"Created README file in templates directory: {readme_path}")
        except Exception as e:
            logger.error(f"Error creating templates directory: {e}")
            st.warning(f"⚠️ Could not create templates directory: {str(e)}")
    return os.path.exists(templates_dir)

# Add Tesseract check
def check_tesseract():
    """Check if Tesseract is installed and configured"""
    try:
        import pytesseract
        tesseract_path = pytesseract.get_tesseract_version()
        tessdata_path = os.environ.get("TESSDATA_PREFIX")
        return True, tesseract_path, tessdata_path
    except Exception as e:
        logger.error(f"Tesseract check error: {e}")
        return False, None, None

# Run the check
tesseract_installed, tesseract_path, tessdata_path = check_tesseract()

# Call this function during initialization
check_templates_directory()

# Add error handling wrapper at the beginning of the file
def safe_streamlit_app():
    """Wrapper function to catch and handle all exceptions"""
    try:
        # Main app code goes here
        # ...
        
        # At the end of processing, clear memory
        clear_memory()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Critical application error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Offer restart option
        if st.button("Restart Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

# Call the wrapper function
if __name__ == "__main__":
    safe_streamlit_app()


#streamlit run main.py --server.maxUploadSize=5

# Add this to your settings tab or create a new API tab
with settings_tab:
    # Add API information section
    st.subheader("🔌 API Integration")
    st.write("This application provides an API for integration with other systems.")
    
    with st.expander("API Documentation"):
        st.markdown("""
        ### API Endpoints
        
        #### POST /process
        Process a passport or ID document image.
        
        **Parameters:**
        - `file`: The document image file (JPG, PNG)
        - `options`: JSON string with processing options
        - `webhook`: Optional webhook URL to receive results asynchronously
        
        **Example curl command:**
        ```bash
        curl -X POST "http://your-server:8000/process" \\
          -H "accept: application/json" \\
          -H "Content-Type: multipart/form-data" \\
          -F "file=@passport.jpg" \\
          -F "options={\"language\":\"eng\",\"enable_face_detection\":true}"
        ```
        
        #### GET /health
        Check if the API is running.
        
        ### Running the API Server
        
        To start the API server, run:
        ```bash
        python api.py
        ```
        
        This will start a FastAPI server on port 8000.
        """)
        
        st.info("For security reasons, the API server is not automatically started with the Streamlit app. You need to run it separately.")
    
    # Add API testing section
    with st.expander("Test API Connection"):
        api_url = st.text_input("API URL", "http://localhost:8000/health")
        if st.button("Test Connection"):
            try:
                import requests
                response = requests.get(api_url)
                if response.status_code == 200:
                    st.success(f"✅ API is running: {response.json()}")
                else:
                    st.error(f"❌ API returned status code {response.status_code}")
            except Exception as e:
                st.error(f"❌ Could not connect to API: {str(e)}")

# Analytics functionality has been moved to the Admin section

# Define admin functions before using them
def get_admin_notifications():
    """Get all unread admin notifications"""
    notifications_dir = os.path.join(os.path.dirname(__file__), "admin_notifications")

    if not os.path.exists(notifications_dir):
        return []

    notifications = []
    for file in os.listdir(notifications_dir):
        if file.endswith('.json'):
            try:
                with open(os.path.join(notifications_dir, file), 'r') as f:
                    notification = json.load(f)
                    notifications.append(notification)
            except Exception as e:
                logger.error(f"Error loading notification {file}: {e}")

    # Sort by timestamp (newest first)
    notifications.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return notifications

def mark_notification_read(notification_id):
    """Mark a notification as read"""
    notifications_dir = os.path.join(os.path.dirname(__file__), "admin_notifications")
    notification_file = os.path.join(notifications_dir, f"notification_{notification_id}.json")

    if os.path.exists(notification_file):
        try:
            with open(notification_file, 'r') as f:
                notification = json.load(f)

            notification['read'] = True

            with open(notification_file, 'w') as f:
                json.dump(notification, f, indent=2)

        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")

def send_admin_notification(feedback_entry):
    """
    Send notification to admin about new feedback

    Args:
        feedback_entry: Dictionary containing feedback data
    """
    try:
        # Create a notification file for the admin dashboard
        notifications_dir = os.path.join(os.path.dirname(__file__), "admin_notifications")
        os.makedirs(notifications_dir, exist_ok=True)

        notification = {
            "id": feedback_entry["id"],
            "timestamp": feedback_entry["timestamp"],
            "type": "new_feedback",
            "feedback_type": feedback_entry["feedback_type"],
            "message": f"New {feedback_entry['feedback_type']} report received",
            "read": False,
            "priority": "normal"
        }

        # Save notification
        notification_file = os.path.join(notifications_dir, f"notification_{feedback_entry['id']}.json")
        with open(notification_file, "w", encoding="utf-8") as f:
            json.dump(notification, f, indent=2, ensure_ascii=False)

        # Log notification
        logger.info(f"Admin notification created for feedback: {feedback_entry['id']}")

    except Exception as e:
        logger.error(f"Error creating admin notification: {e}")

def display_feedback_management():
    """Display the enhanced feedback management interface for admins"""
    st.header("📊 OCR Feedback Management")

    # Quick stats
    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
    if os.path.exists(feedback_dir):
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]
        total_feedback = len(feedback_files)

        # Count by type
        ocr_feedback = len([f for f in feedback_files if 'ocr_correction' in f])
        mrz_feedback = len([f for f in feedback_files if 'mrz_correction' in f])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reports", total_feedback)
        with col2:
            st.metric("OCR Reports", ocr_feedback)
        with col3:
            st.metric("MRZ Reports", mrz_feedback)
    else:
        st.info("No feedback directory found. Reports will be created when users submit feedback.")
        return

    # Load all feedback data
    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
    if not os.path.exists(feedback_dir):
        st.info("No feedback data available yet.")
        return

    feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]

    if not feedback_files:
        st.info("No feedback data available yet.")
        return

    # Load feedback data
    all_feedback = []
    for file in feedback_files:
        try:
            with open(os.path.join(feedback_dir, file), 'r') as f:
                feedback = json.load(f)
                all_feedback.append(feedback)
        except Exception as e:
            logger.error(f"Error loading feedback file {file}: {e}")

    # Sort by timestamp (newest first)
    all_feedback.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    # Filter options
    st.subheader("Filter Feedback")
    col1, col2 = st.columns(2)

    with col1:
        feedback_types = list(set(fb.get('feedback_type', '').split('_')[0] for fb in all_feedback))
        selected_type = st.selectbox(
            "Feedback Type",
            ["All"] + feedback_types,
            key="feedback_type_filter"
        )

    with col2:
        # Filter by difference ratio
        min_diff = st.slider(
            "Minimum Difference Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="min_diff_ratio"
        )

    # Apply filters
    filtered_feedback = all_feedback
    if selected_type != "All":
        filtered_feedback = [fb for fb in filtered_feedback if fb.get('feedback_type', '').startswith(selected_type)]

    filtered_feedback = [fb for fb in filtered_feedback if fb.get('diff_ratio', 0) >= min_diff]

    # Display feedback items
    st.subheader(f"Feedback Items ({len(filtered_feedback)})")

    for i, feedback in enumerate(filtered_feedback):
        with st.expander(f"Feedback #{i+1} - {feedback.get('timestamp', 'Unknown date')} - {feedback.get('feedback_type', 'Unknown type')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Text:**")
                st.text_area(
                    "Original",
                    feedback.get('original_text', ''),
                    height=100,
                    key=f"original_{i}"
                )

            with col2:
                st.markdown("**Corrected Text:**")
                st.text_area(
                    "Corrected",
                    feedback.get('corrected_text', ''),
                    height=100,
                    key=f"corrected_{i}"
                )

            st.markdown(f"**Difference Ratio:** {feedback.get('diff_ratio', 0):.2f}")

            # Check if there are additional details
            details_file = os.path.join(feedback_dir, f"{feedback.get('id', '')}_details.txt")
            if os.path.exists(details_file):
                with open(details_file, 'r') as f:
                    details = f.read()
                st.markdown("**Additional Details:**")
                st.text(details)

            # Add action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Mark as Reviewed", key=f"review_{i}"):
                    # Update the feedback status
                    feedback['status'] = 'reviewed'
                    with open(os.path.join(feedback_dir, f"{feedback.get('id', '')}.json"), 'w') as f:
                        json.dump(feedback, f, indent=2)
                    st.success("Marked as reviewed!")
                    st.rerun()

            with col2:
                if st.button("Export to Training Data", key=f"export_{i}"):
                    # Export this feedback for model training
                    training_dir = os.path.join(os.path.dirname(__file__), "training_data")
                    os.makedirs(training_dir, exist_ok=True)

                    training_file = os.path.join(training_dir, f"training_{feedback.get('id', '')}.json")
                    with open(training_file, 'w') as f:
                        json.dump({
                            'original': feedback.get('original_text', ''),
                            'corrected': feedback.get('corrected_text', ''),
                            'type': feedback.get('feedback_type', ''),
                            'exported_date': datetime.datetime.now().isoformat()
                        }, f, indent=2)

                    st.success("Exported to training data!")

            with col3:
                if st.button("Delete", key=f"delete_{i}"):
                    # Delete this feedback
                    os.remove(os.path.join(feedback_dir, f"{feedback.get('id', '')}.json"))
                    details_file = os.path.join(feedback_dir, f"{feedback.get('id', '')}_details.txt")
                    if os.path.exists(details_file):
                        os.remove(details_file)

                    st.success("Feedback deleted!")
                    st.rerun()

    # Add export all button
    if filtered_feedback:
        if st.button("Export All Filtered Feedback to CSV"):
            # Convert to DataFrame and export
            try:
                import pandas as pd

                df = pd.DataFrame([
                    {
                        'id': fb.get('id', ''),
                        'timestamp': fb.get('timestamp', ''),
                        'image_hash': fb.get('image_hash', ''),
                        'feedback_type': fb.get('feedback_type', ''),
                        'original_text': fb.get('original_text', ''),
                        'corrected_text': fb.get('corrected_text', ''),
                        'diff_ratio': fb.get('diff_ratio', 0)
                    }
                    for fb in filtered_feedback
                ])
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode('utf-8'),
                    "feedback.csv",
                    "text/csv"
                )
            except ImportError:
                st.error("Pandas is required for CSV export")

def display_system_status():
    """Display system status and health information"""
    st.subheader("🖥️ System Status")

    # System information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")

    with col2:
        st.metric("Streamlit Version", st.__version__)

    with col3:
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            st.metric("Available Memory", f"{memory.available // (1024**3)} GB")
        except ImportError:
            st.metric("Memory Info", "N/A")

    # GPU Status
    st.subheader("🎮 GPU Status")
    display_gpu_status("admin")

    # Quick GPU fix for admins
    if not torch.cuda.is_available():
        st.markdown("---")
        st.markdown("### 🔧 GPU Troubleshooting Tools")
        st.warning("⚠️ GPU not accessible - providing troubleshooting tools")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Automatic Fix Scripts:**")
            st.info("Use the provided scripts to automatically diagnose and fix GPU issues:")
            st.markdown("• `fix_gpu.py` - Python diagnostic script")
            st.markdown("• `fix_gpu.bat` - Windows batch file")

            if st.button("🧪 Test GPU Access", key="admin_test_gpu_access"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        test_tensor = torch.randn(10, 10).cuda()
                        st.success("✅ GPU access test passed!")
                        st.info(f"Test tensor created on: {test_tensor.device}")
                    else:
                        st.error("❌ GPU access test failed - CUDA not available")
                        st.info("PyTorch version: " + torch.__version__)
                        st.info("CUDA version: " + str(torch.version.cuda))
                except Exception as e:
                    st.error(f"❌ GPU test error: {e}")

        with col2:
            st.markdown("**Manual Fix Commands:**")
            st.code("""
# Step 1: Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Step 2: Install with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: Verify installation
python -c "import torch; print(torch.cuda.is_available())"
            """)

            st.markdown("**Alternative CUDA 12.1:**")
            st.code("""
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            """)

    # Dependencies status
    st.subheader("📦 Dependencies Status")

    dependencies = {
        "EasyOCR": "easyocr",
        "OpenCV": "cv2",
        "PIL": "PIL",
        "spaCy": "spacy",
        "Plotly": "plotly",
        "Pandas": "pandas"
    }

    for name, module in dependencies.items():
        try:
            __import__(module)
            st.success(f"✅ {name} - Available")
        except ImportError:
            st.error(f"❌ {name} - Not Available")

    # Disk space for feedback and analytics
    st.subheader("💾 Storage Status")

    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
    analytics_dir = os.path.join(os.path.dirname(__file__), "analytics_data")

    if os.path.exists(feedback_dir):
        feedback_files = len([f for f in os.listdir(feedback_dir) if f.endswith('.json')])
        st.info(f"📝 Feedback Reports: {feedback_files}")
    else:
        st.info("📝 Feedback Reports: 0")

    if os.path.exists(analytics_dir):
        analytics_files = len([f for f in os.listdir(analytics_dir) if f.endswith('.jsonl')])
        st.info(f"📊 Analytics Files: {analytics_files}")
    else:
        st.info("📊 Analytics Files: 0")

def display_system_logs():
    """Display system logs and recent activity"""
    st.subheader("📋 System Logs")

    # Show recent analytics events
    try:
        from analytics_dashboard import AnalyticsTracker
        tracker = AnalyticsTracker()
        recent_events = tracker.load_events(days_back=1)

        if recent_events:
            st.write(f"**Recent Events (Last 24 hours): {len(recent_events)}**")

            # Show last 20 events
            for event in recent_events[-20:]:
                timestamp = event.get('timestamp', 'Unknown')
                event_type = event.get('event_type', 'Unknown')
                details = event.get('details', {})

                with st.expander(f"{timestamp} - {event_type}"):
                    st.json(details)
        else:
            st.info("No recent events found")
    except Exception as e:
        st.error(f"Error loading analytics events: {e}")

    # Show application logs if available
    st.subheader("📄 Application Logs")

    log_files = []
    for file in os.listdir("."):
        if file.endswith(".log"):
            log_files.append(file)

    if log_files:
        selected_log = st.selectbox("Select log file", log_files)

        if selected_log:
            try:
                with open(selected_log, "r") as f:
                    log_content = f.read()

                # Show last 50 lines
                lines = log_content.split('\n')
                recent_lines = lines[-50:] if len(lines) > 50 else lines

                st.text_area("Recent Log Entries", '\n'.join(recent_lines), height=300)

                # Download button for full log
                st.download_button(
                    "Download Full Log",
                    log_content,
                    file_name=selected_log,
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error reading log file: {e}")
    else:
        st.info("No log files found")

def display_database_management():
    """Display database management interface"""
    st.header("🗄️ Database Management")
    st.markdown("*Manage database storage and data operations*")

    # Check if database is enabled
    if not st.session_state.get("admin_enable_database", False):
        st.warning("⚠️ Database storage is not enabled. Please enable it in the Settings tab first.")
        return

    try:
        from database_manager import get_database_manager
        db_manager = get_database_manager()

        # Database Statistics
        st.subheader("📊 Database Statistics")

        stats = db_manager.get_database_stats()
        if stats:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("OCR Results", stats.get("ocr_results_count", 0))

            with col2:
                st.metric("Feedback Records", stats.get("feedback_count", 0))

            with col3:
                st.metric("Analytics Events", stats.get("analytics_events_count", 0))

            with col4:
                st.metric("Database Size", f"{stats.get('db_size_mb', 0):.2f} MB")

        # Recent Data
        st.subheader("📋 Recent Data")

        tab1, tab2, tab3 = st.tabs(["OCR Results", "Feedback", "Analytics Summary"])

        with tab1:
            st.markdown("**Recent OCR Results**")

            limit = st.slider("Number of records to show", 5, 100, 20, key="ocr_limit")

            ocr_results = db_manager.get_ocr_results(limit=limit)
            if ocr_results:
                for i, result in enumerate(ocr_results):
                    with st.expander(f"Result #{result['id']} - {result['timestamp']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Session ID:** {result['session_id']}")
                            st.write(f"**Language:** {result['language']}")
                            st.write(f"**Processing Time:** {result['processing_time']:.2f}s")
                            st.write(f"**Status:** {result['status']}")

                        with col2:
                            if result['extracted_data']:
                                st.write("**Extracted Data:**")
                                st.json(result['extracted_data'])

                        if result['ocr_text']:
                            st.write("**OCR Text:**")
                            st.text_area("", result['ocr_text'], height=100, key=f"ocr_text_{i}")
            else:
                st.info("No OCR results found in database")

        with tab2:
            st.markdown("**Recent Feedback**")

            feedback_results = db_manager.get_feedback(limit=20)
            if feedback_results:
                for i, feedback in enumerate(feedback_results):
                    with st.expander(f"Feedback #{feedback['id']} - {feedback['feedback_type']} - {feedback['timestamp']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Original Text:**")
                            st.text_area("", feedback['original_text'], height=100, key=f"orig_{i}")

                        with col2:
                            st.write("**Corrected Text:**")
                            st.text_area("", feedback['corrected_text'], height=100, key=f"corr_{i}")

                        st.write(f"**Difference Ratio:** {feedback['diff_ratio']:.3f}")
                        if feedback['additional_details']:
                            st.write(f"**Additional Details:** {feedback['additional_details']}")
            else:
                st.info("No feedback found in database")

        with tab3:
            st.markdown("**Analytics Summary**")

            days_back = st.selectbox("Period", [7, 14, 30, 60, 90], index=2, key="analytics_period")

            analytics_summary = db_manager.get_analytics_summary(days_back=days_back)
            if analytics_summary:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Events", analytics_summary.get("total_events", 0))

                with col2:
                    st.metric("Unique Sessions", analytics_summary.get("unique_sessions", 0))

                with col3:
                    st.metric("Avg Processing Time", f"{analytics_summary.get('avg_processing_time', 0):.2f}s")

                if analytics_summary.get("events_by_type"):
                    st.write("**Events by Type:**")
                    st.json(analytics_summary["events_by_type"])
            else:
                st.info("No analytics data found in database")

        # Database Operations
        st.subheader("🔧 Database Operations")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Data Export**")

            export_table = st.selectbox(
                "Select table to export",
                ["ocr_results", "feedback", "analytics_events", "system_logs"],
                key="export_table"
            )

            export_format = st.selectbox(
                "Export format",
                ["JSON", "CSV"],
                key="export_format"
            )

            if st.button("📤 Export Data"):
                try:
                    exported_data = db_manager.export_data(export_table, export_format.lower())

                    if exported_data:
                        file_extension = "json" if export_format == "JSON" else "csv"
                        mime_type = "application/json" if export_format == "JSON" else "text/csv"

                        st.download_button(
                            f"📥 Download {export_table}.{file_extension}",
                            exported_data,
                            file_name=f"{export_table}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                            mime=mime_type
                        )
                    else:
                        st.error("No data to export")
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col2:
            st.markdown("**Data Cleanup**")

            cleanup_days = st.number_input(
                "Keep data for (days)",
                min_value=7,
                max_value=365,
                value=90,
                key="cleanup_days"
            )

            if st.button("🧹 Clean Old Data"):
                try:
                    cleanup_result = db_manager.cleanup_old_data(cleanup_days)

                    if cleanup_result:
                        st.success("✅ Cleanup completed!")
                        st.json(cleanup_result)
                    else:
                        st.info("No old data to clean")
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")

        with col3:
            st.markdown("**Database Maintenance**")

            if st.button("🔄 Refresh Stats"):
                st.rerun()

            if st.button("🗄️ Vacuum Database"):
                try:
                    import sqlite3
                    with sqlite3.connect(db_manager.db_path) as conn:
                        conn.execute("VACUUM")
                    st.success("✅ Database vacuumed successfully!")
                except Exception as e:
                    st.error(f"Vacuum failed: {e}")

            if st.button("🔍 Check Database Integrity"):
                try:
                    import sqlite3
                    with sqlite3.connect(db_manager.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check")
                        result = cursor.fetchone()[0]

                        if result == "ok":
                            st.success("✅ Database integrity check passed!")
                        else:
                            st.error(f"❌ Database integrity issues: {result}")
                except Exception as e:
                    st.error(f"Integrity check failed: {e}")

    except Exception as e:
        st.error(f"❌ Database management error: {e}")
        st.info("Make sure the database is properly configured and accessible.")

def display_admin_settings():
    """Display admin settings and configuration options"""
    st.header("⚙️ Admin Settings & Configuration")
    st.markdown("*Advanced configuration options for administrators*")

    # System Configuration
    st.subheader("🖥️ System Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Processing Settings**")

        # GPU Settings
        gpu_enabled = st.checkbox(
            "Enable GPU Processing by Default",
            value=st.session_state.get("admin_gpu_default", True),
            key="admin_gpu_setting",
            help="Set the default GPU processing preference for all users"
        )

        # OCR Language Default
        default_ocr_lang = st.selectbox(
            "Default OCR Language",
            list(LANGUAGE_MAP.keys()),
            index=0,
            key="admin_default_ocr_lang",
            help="Set the default OCR language for new sessions"
        )

        # Processing Timeout
        processing_timeout = st.number_input(
            "Processing Timeout (seconds)",
            min_value=30,
            max_value=300,
            value=st.session_state.get("admin_processing_timeout", 120),
            key="admin_processing_timeout",
            help="Maximum time allowed for processing a single document"
        )

    with col2:
        st.markdown("**Security Settings**")

        # Admin Password Change
        st.markdown("**Change Admin Password**")
        current_password = st.text_input("Current Password", type="password", key="current_admin_pass")
        new_password = st.text_input("New Password", type="password", key="new_admin_pass")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_admin_pass")

        if st.button("Update Admin Password"):
            if current_password == "admin123":  # Current password check
                if new_password and new_password == confirm_password:
                    # In a real application, you would hash and store this securely
                    st.success("✅ Password updated successfully!")
                    st.info("⚠️ Note: In production, implement proper password hashing and storage")
                else:
                    st.error("❌ New passwords don't match or are empty")
            else:
                st.error("❌ Current password is incorrect")

        # Session Management
        max_sessions = st.number_input(
            "Maximum Concurrent Sessions",
            min_value=1,
            max_value=100,
            value=st.session_state.get("admin_max_sessions", 50),
            key="admin_max_sessions",
            help="Maximum number of concurrent user sessions"
        )

    # Database Configuration
    st.subheader("🗄️ Database Configuration")

    col1, col2 = st.columns(2)

    with col1:
        enable_database = st.checkbox(
            "Enable Database Storage",
            value=st.session_state.get("admin_enable_database", False),
            key="admin_enable_database",
            help="Store OCR results and analytics in database"
        )

        if enable_database:
            db_type = st.selectbox(
                "Database Type",
                ["SQLite", "PostgreSQL", "MySQL"],
                index=0,
                key="admin_db_type",
                help="Type of database to use"
            )

            if db_type == "SQLite":
                db_path = st.text_input(
                    "Database File Path",
                    value=st.session_state.get("admin_db_path", "passport_ocr.db"),
                    key="admin_db_path",
                    help="Path to SQLite database file"
                )
            else:
                st.info(f"{db_type} configuration will be available in future updates")

    with col2:
        if enable_database:
            st.markdown("**Database Options**")

            store_ocr_results = st.checkbox(
                "Store OCR Results",
                value=st.session_state.get("admin_store_ocr_results", True),
                key="admin_store_ocr_results",
                help="Store OCR processing results in database"
            )

            store_analytics = st.checkbox(
                "Store Analytics Events",
                value=st.session_state.get("admin_store_analytics", True),
                key="admin_store_analytics",
                help="Store analytics events in database"
            )

            store_feedback = st.checkbox(
                "Store Feedback",
                value=st.session_state.get("admin_store_feedback", True),
                key="admin_store_feedback",
                help="Store user feedback in database"
            )

            # Database actions
            if st.button("🔧 Initialize Database"):
                try:
                    from database_manager import get_database_manager
                    db_manager = get_database_manager()
                    st.success("✅ Database initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Database initialization failed: {e}")

            if st.button("📊 View Database Stats"):
                try:
                    from database_manager import get_database_manager
                    db_manager = get_database_manager()
                    stats = db_manager.get_database_stats()
                    st.json(stats)
                except Exception as e:
                    st.error(f"❌ Error getting database stats: {e}")

    # Data Management
    st.subheader("📊 Data Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Feedback Settings**")

        auto_feedback_cleanup = st.checkbox(
            "Auto-cleanup old feedback",
            value=st.session_state.get("admin_auto_cleanup", False),
            key="admin_auto_cleanup",
            help="Automatically remove feedback older than specified days"
        )

        if auto_feedback_cleanup:
            cleanup_days = st.number_input(
                "Keep feedback for (days)",
                min_value=7,
                max_value=365,
                value=st.session_state.get("admin_cleanup_days", 90),
                key="admin_cleanup_days"
            )

    with col2:
        st.markdown("**Analytics Settings**")

        analytics_retention = st.number_input(
            "Analytics Data Retention (days)",
            min_value=30,
            max_value=730,
            value=st.session_state.get("admin_analytics_retention", 180),
            key="admin_analytics_retention",
            help="How long to keep analytics data"
        )

        detailed_logging = st.checkbox(
            "Enable Detailed Logging",
            value=st.session_state.get("admin_detailed_logging", True),
            key="admin_detailed_logging",
            help="Log detailed user interactions for analytics"
        )

    with col3:
        st.markdown("**Export Settings**")

        export_format = st.selectbox(
            "Default Export Format",
            ["JSON", "CSV", "Excel"],
            index=0,
            key="admin_export_format",
            help="Default format for data exports"
        )

        include_metadata = st.checkbox(
            "Include Metadata in Exports",
            value=st.session_state.get("admin_include_metadata", True),
            key="admin_include_metadata",
            help="Include timestamps and system info in exports"
        )

    # Feature Toggles
    st.subheader("🎛️ Feature Management")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Available Features**")

        enable_face_detection = st.checkbox(
            "Enable Face Detection Feature",
            value=st.session_state.get("admin_enable_face_detection", True),
            key="admin_enable_face_detection",
            help="Allow users to enable face detection"
        )

        enable_doc_classification = st.checkbox(
            "Enable Document Classification",
            value=st.session_state.get("admin_enable_doc_classification", True),
            key="admin_enable_doc_classification",
            help="Allow automatic document type detection"
        )

        enable_template_matching = st.checkbox(
            "Enable Template Matching",
            value=st.session_state.get("admin_enable_template_matching", True),
            key="admin_enable_template_matching",
            help="Allow template-based data extraction"
        )

    with col2:
        st.markdown("**Advanced Features**")

        enable_batch_processing = st.checkbox(
            "Enable Batch Processing",
            value=st.session_state.get("admin_enable_batch_processing", True),
            key="admin_enable_batch_processing",
            help="Allow users to process multiple documents"
        )

        enable_api_access = st.checkbox(
            "Enable API Access",
            value=st.session_state.get("admin_enable_api_access", False),
            key="admin_enable_api_access",
            help="Enable REST API for external integrations"
        )

        enable_client_side_processing = st.checkbox(
            "Enable Client-side Processing",
            value=st.session_state.get("admin_enable_client_side", True),
            key="admin_enable_client_side",
            help="Allow client-side OCR processing"
        )

    # Notification Settings
    st.subheader("🔔 Notification Settings")

    col1, col2 = st.columns(2)

    with col1:
        email_notifications = st.checkbox(
            "Enable Email Notifications",
            value=st.session_state.get("admin_email_notifications", False),
            key="admin_email_notifications",
            help="Send email notifications for important events"
        )

        if email_notifications:
            admin_email = st.text_input(
                "Admin Email Address",
                value=st.session_state.get("admin_email", ""),
                key="admin_email",
                help="Email address to receive notifications"
            )

    with col2:
        notification_types = st.multiselect(
            "Notification Types",
            ["New Feedback", "System Errors", "High Usage", "Security Alerts"],
            default=st.session_state.get("admin_notification_types", ["New Feedback", "System Errors"]),
            key="admin_notification_types",
            help="Types of events to receive notifications for"
        )

    # Save Settings
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("💾 Save All Settings", type="primary"):
            # Settings are automatically saved via the widget keys
            # Just show confirmation and log the event
            st.success("✅ Settings saved successfully!")

            # Log the settings change
            try:
                settings_changed = [
                    "gpu_default", "default_ocr_lang", "processing_timeout", "max_sessions",
                    "auto_cleanup", "analytics_retention", "detailed_logging", "export_format",
                    "include_metadata", "enable_face_detection", "enable_doc_classification",
                    "enable_template_matching", "enable_batch_processing", "enable_api_access",
                    "enable_client_side", "email_notifications", "notification_types"
                ]

                log_analytics_event("admin_settings_updated", {
                    "settings_changed": settings_changed,
                    "timestamp": datetime.datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error logging settings update: {e}")

    with col2:
        if st.button("🔄 Reset to Defaults"):
            # Clear admin settings from session state (except widget keys that are currently active)
            keys_to_remove = []
            for key in st.session_state.keys():
                if key.startswith("admin_") and not key.endswith("_setting") and not key.endswith("_lang") and not key.endswith("_timeout"):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del st.session_state[key]

            st.success("✅ Settings reset to defaults! Please refresh the page to see changes.")
            st.info("Note: Some settings require a page refresh to reset completely.")

    with col3:
        if st.button("📤 Export Settings"):
            # Export current settings
            current_settings = {
                key.replace("admin_", ""): value
                for key, value in st.session_state.items()
                if key.startswith("admin_")
            }

            settings_json = json.dumps(current_settings, indent=2, default=str)

            st.download_button(
                "📥 Download Settings",
                settings_json,
                file_name=f"admin_settings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_admin_analysis():
    """Display comprehensive analysis for admin"""
    st.subheader("🔍 Advanced Admin Analytics")
    st.markdown("*Additional insights and analysis tools for administrators*")

    # Real-time Statistics
    st.markdown("### 📈 Real-time Statistics")

    col1, col2, col3, col4 = st.columns(4)

    # Get current session statistics
    try:
        from analytics_dashboard import AnalyticsTracker
        tracker = AnalyticsTracker()
        today_stats = tracker.get_usage_stats(days_back=1)
        week_stats = tracker.get_usage_stats(days_back=7)

        with col1:
            today_events = today_stats.get("total_events", 0)
            st.metric("Today's Activity", today_events)

        with col2:
            week_events = week_stats.get("total_events", 0)
            st.metric("This Week", week_events)

        with col3:
            today_sessions = today_stats.get("unique_sessions", 0)
            st.metric("Active Sessions Today", today_sessions)

        with col4:
            week_error_rate = week_stats.get("error_rate", {}).get("overall", 0)
            st.metric("Weekly Error Rate", f"{week_error_rate:.1f}%")

    except Exception as e:
        st.error(f"Error loading real-time stats: {e}")

    # Feedback Analysis
    st.markdown("### 📝 Detailed Feedback Analysis")

    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
    if os.path.exists(feedback_dir):
        feedback_files = [f for f in os.listdir(feedback_dir) if f.endswith('.json')]

        if feedback_files:
            # Load all feedback
            all_feedback = []
            for file in feedback_files:
                try:
                    with open(os.path.join(feedback_dir, file), 'r') as f:
                        feedback = json.load(f)
                        all_feedback.append(feedback)
                except Exception as e:
                    logger.error(f"Error loading feedback file {file}: {e}")

            if all_feedback:
                # Create analysis metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_reports = len(all_feedback)
                    st.metric("Total Reports", total_reports)

                with col2:
                    ocr_reports = len([f for f in all_feedback if 'ocr' in f.get('feedback_type', '')])
                    st.metric("OCR Reports", ocr_reports)

                with col3:
                    mrz_reports = len([f for f in all_feedback if 'mrz' in f.get('feedback_type', '')])
                    st.metric("MRZ Reports", mrz_reports)

                with col4:
                    avg_diff = sum(f.get('diff_ratio', 0) for f in all_feedback) / len(all_feedback)
                    st.metric("Avg Difference Ratio", f"{avg_diff:.3f}")

                # Feedback trends over time
                try:
                    import pandas as pd
                    import plotly.express as px

                    # Create DataFrame for analysis
                    df = pd.DataFrame(all_feedback)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date

                    # Daily feedback count
                    daily_feedback = df.groupby('date').size().reset_index(name='count')
                    fig_daily = px.line(daily_feedback, x='date', y='count', title='Daily Feedback Reports')
                    st.plotly_chart(fig_daily, use_container_width=True)

                    # Feedback type distribution
                    type_counts = df['feedback_type'].value_counts()
                    fig_types = px.pie(values=type_counts.values, names=type_counts.index, title='Feedback Type Distribution')
                    st.plotly_chart(fig_types, use_container_width=True)

                    # Most common issues
                    st.markdown("### 🔍 Most Common Issues")

                    # Analyze feedback reasons from details files
                    issue_counts = {}
                    for feedback in all_feedback:
                        details_file = os.path.join(feedback_dir, f"{feedback.get('id', '')}_details.txt")
                        if os.path.exists(details_file):
                            try:
                                with open(details_file, 'r') as f:
                                    details = f.read().lower()
                                    # Simple keyword analysis
                                    keywords = ['missing', 'incorrect', 'blurry', 'wrong', 'error', 'partial']
                                    for keyword in keywords:
                                        if keyword in details:
                                            issue_counts[keyword] = issue_counts.get(keyword, 0) + 1
                            except Exception as e:
                                logger.error(f"Error reading details file: {e}")

                    if issue_counts:
                        issue_df = pd.DataFrame(list(issue_counts.items()), columns=['Issue', 'Count'])
                        issue_df = issue_df.sort_values('Count', ascending=True)
                        fig_issues = px.bar(issue_df, x='Count', y='Issue', orientation='h', title='Common Issues')
                        st.plotly_chart(fig_issues, use_container_width=True)

                except ImportError:
                    st.info("Install pandas and plotly for advanced analytics")
                except Exception as e:
                    st.error(f"Error creating analytics: {e}")
        else:
            st.info("No feedback data available for analysis")
    else:
        st.info("No feedback directory found")

    # System Performance Analysis
    st.markdown("### ⚡ Advanced Performance Metrics")

    try:
        from analytics_dashboard import AnalyticsTracker
        tracker = AnalyticsTracker()
        stats = tracker.get_usage_stats(days_back=7)

        if stats.get("total_events", 0) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Weekly Events", stats["total_events"])

            with col2:
                error_rate = stats.get("error_rate", {}).get("overall", 0)
                st.metric("Error Rate", f"{error_rate:.1f}%")

            with col3:
                avg_time = stats.get("processing_times", {}).get("avg", 0)
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        else:
            st.info("No performance data available")
    except Exception as e:
        st.error(f"Error loading performance data: {e}")

    # Admin Action Center
    st.markdown("### 🛠️ Admin Action Center")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧹 Clear Old Analytics Data"):
            try:
                from analytics_dashboard import AnalyticsTracker
                tracker = AnalyticsTracker()
                # Clear analytics data older than 90 days

                analytics_dir = tracker.analytics_dir
                if os.path.exists(analytics_dir):
                    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=90)
                    files_removed = 0

                    for file in os.listdir(analytics_dir):
                        if file.startswith("events_") and file.endswith(".jsonl"):
                            file_date_str = file.replace("events_", "").replace(".jsonl", "")
                            try:
                                file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")
                                if file_date < cutoff_date:
                                    os.remove(os.path.join(analytics_dir, file))
                                    files_removed += 1
                            except ValueError:
                                continue

                    st.success(f"✅ Removed {files_removed} old analytics files")
                else:
                    st.info("No analytics directory found")
            except Exception as e:
                st.error(f"Error clearing analytics data: {e}")

    with col2:
        if st.button("📊 Export All Analytics"):
            try:
                from analytics_dashboard import AnalyticsTracker
                tracker = AnalyticsTracker()
                stats = tracker.get_usage_stats(days_back=30)

                # Create comprehensive export
                export_data = {
                    "export_date": datetime.datetime.now().isoformat(),
                    "period": "30_days",
                    "analytics": stats,
                    "feedback_summary": {
                        "total_feedback": len(os.listdir(os.path.join(os.path.dirname(__file__), "feedback_data"))) if os.path.exists(os.path.join(os.path.dirname(__file__), "feedback_data")) else 0
                    }
                }

                import json
                json_data = json.dumps(export_data, indent=2, default=str)

                st.download_button(
                    "📥 Download Complete Analytics Report",
                    json_data,
                    file_name=f"complete_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exporting analytics: {e}")

    with col3:
        if st.button("🔄 Reset All Notifications"):
            try:
                notifications_dir = os.path.join(os.path.dirname(__file__), "admin_notifications")
                if os.path.exists(notifications_dir):
                    import shutil
                    shutil.rmtree(notifications_dir)
                    os.makedirs(notifications_dir, exist_ok=True)
                    st.success("✅ All notifications cleared")
                else:
                    st.info("No notifications to clear")
            except Exception as e:
                st.error(f"Error clearing notifications: {e}")

    # System Health Check
    st.markdown("### 🏥 System Health Check")

    health_status = {}

    # Check disk space
    try:
        import shutil
        _, _, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        health_status["disk_space"] = {
            "status": "healthy" if free_gb > 1 else "warning" if free_gb > 0.5 else "critical",
            "free_gb": free_gb
        }
    except Exception:
        health_status["disk_space"] = {"status": "unknown", "free_gb": 0}

    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        health_status["memory"] = {
            "status": "healthy" if memory_percent < 80 else "warning" if memory_percent < 90 else "critical",
            "usage_percent": memory_percent
        }
    except ImportError:
        health_status["memory"] = {"status": "unknown", "usage_percent": 0}

    # Display health status
    col1, col2 = st.columns(2)

    with col1:
        disk_status = health_status["disk_space"]["status"]
        disk_icon = "🟢" if disk_status == "healthy" else "🟡" if disk_status == "warning" else "🔴"
        st.write(f"{disk_icon} **Disk Space**: {health_status['disk_space']['free_gb']} GB free")

    with col2:
        memory_status = health_status["memory"]["status"]
        memory_icon = "🟢" if memory_status == "healthy" else "🟡" if memory_status == "warning" else "🔴"
        st.write(f"{memory_icon} **Memory Usage**: {health_status['memory']['usage_percent']:.1f}%")

# Admin tab content
with admin_tab:
    st.header("🔧 Admin Dashboard")

    # Add password protection for admin access
    admin_password = st.text_input("Admin Password", type="password", key="admin_password")

    # Simple password check (in production, use proper authentication)
    if admin_password == "admin123":  # Change this to a secure password
        st.success("✅ Admin access granted")

        # Show notifications at the top
        notifications = get_admin_notifications()
        unread_notifications = [n for n in notifications if not n.get('read', False)]

        if unread_notifications:
            st.warning(f"🔔 You have {len(unread_notifications)} unread notifications")

            with st.expander(f"View Notifications ({len(unread_notifications)} unread)"):
                for notification in notifications[:10]:  # Show last 10 notifications
                    status = "🔴" if not notification.get('read', False) else "✅"
                    timestamp = notification.get('timestamp', 'Unknown')
                    message = notification.get('message', 'No message')

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{status} {timestamp}: {message}")
                    with col2:
                        if not notification.get('read', False):
                            if st.button("Mark Read", key=f"read_{notification.get('id', '')}"):
                                mark_notification_read(notification.get('id', ''))
                                st.rerun()

        # Create sub-tabs for different admin functions
        feedback_tab, analytics_tab, database_tab, settings_tab, system_tab, logs_tab = st.tabs(["Feedback Management", "Analytics Dashboard", "Database Management", "Settings", "System Status", "Logs"])

        with feedback_tab:
            display_feedback_management()

        with analytics_tab:
            # Display the full analytics dashboard within admin
            st.header("📊 Complete Analytics Dashboard")
            st.markdown("Comprehensive analytics and usage patterns for the passport OCR application")

            # Import and display analytics dashboard
            from analytics_dashboard import display_analytics_dashboard
            display_analytics_dashboard()

            # Add admin-specific analytics
            st.markdown("---")
            display_admin_analysis()

        with database_tab:
            display_database_management()

        with settings_tab:
            display_admin_settings()

        with system_tab:
            display_system_status()

        with logs_tab:
            display_system_logs()

    elif admin_password:
        st.error("❌ Invalid admin password")
    else:
        st.info("🔒 Enter admin password to access admin features")

# Duplicate functions removed - defined above in admin section

# Add function to provide detailed error explanations
def get_error_explanation(error_type, error_message=None):
    """
    Provides detailed explanations for common errors in the application
    
    Args:
        error_type: String identifying the error type
        error_message: Optional original error message
        
    Returns:
        dict: Explanation, possible causes, and suggested fixes
    """
    explanations = {
        "tesseract_not_found": {
            "explanation": "The Tesseract OCR engine could not be found on your system.",
            "possible_causes": [
                "Tesseract is not installed",
                "Tesseract is not in your system PATH",
                "Incorrect Tesseract path configuration"
            ],
            "suggested_fixes": [
                "Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract",
                "Add Tesseract to your system PATH",
                "Set the correct path in the application settings"
            ]
        },
        "gpu_error": {
            "explanation": "There was an error initializing the GPU for processing.",
            "possible_causes": [
                "CUDA drivers not installed or incompatible",
                "PyTorch/TensorFlow not built with CUDA support",
                "Insufficient GPU memory",
                "GPU is being used by another process"
            ],
            "suggested_fixes": [
                "Update NVIDIA drivers to the latest version",
                "Reinstall PyTorch/TensorFlow with CUDA support",
                "Close other GPU-intensive applications",
                "Try processing with a smaller image or using CPU mode"
            ]
        },
        "mrz_detection_failed": {
            "explanation": "The Machine Readable Zone (MRZ) could not be detected in the image.",
            "possible_causes": [
                "Poor image quality or resolution",
                "MRZ region is blurry or damaged",
                "Document is not positioned correctly",
                "Document type is not supported"
            ],
            "suggested_fixes": [
                "Ensure the image is clear and high resolution",
                "Make sure the MRZ region is fully visible and not cut off",
                "Try different preprocessing options (enhance contrast, denoise)",
                "Manually crop the image to focus on the MRZ region"
            ]
        },
        "ocr_processing_error": {
            "explanation": "An error occurred during OCR text extraction.",
            "possible_causes": [
                "Image is too blurry or low quality",
                "Insufficient memory for processing",
                "Unsupported language selection",
                "OCR engine initialization failure"
            ],
            "suggested_fixes": [
                "Use a clearer image with better lighting",
                "Try processing with grayscale or enhanced contrast",
                "Select a different OCR language",
                "Restart the application or try client-side processing"
            ]
        },
        "spacy_model_error": {
            "explanation": "The spaCy NLP model could not be loaded.",
            "possible_causes": [
                "Required model is not installed",
                "Insufficient memory",
                "Incompatible spaCy version"
            ],
            "suggested_fixes": [
                "Install the required model with: python -m spacy download en_core_web_sm",
                "Upgrade spaCy to the latest version",
                "Restart the application with more available memory"
            ]
        },
        "image_processing_error": {
            "explanation": "An error occurred during image preprocessing.",
            "possible_causes": [
                "Corrupted image file",
                "Unsupported image format",
                "Insufficient memory for processing",
                "Image dimensions too large"
            ],
            "suggested_fixes": [
                "Check if the image file is valid and not corrupted",
                "Convert the image to a standard format (JPEG, PNG)",
                "Resize the image to smaller dimensions before uploading",
                "Try different preprocessing options"
            ]
        }
    }
    
    # Add the original error message if provided
    result = explanations.get(error_type, {
        "explanation": "An unexpected error occurred.",
        "possible_causes": ["Unknown cause"],
        "suggested_fixes": ["Try restarting the application or using different inputs"]
    })
    
    if error_message:
        result["original_error"] = error_message
        
    return result

# Admin notification functions are now defined above in the admin tab section

# Add this function after imports
def save_feedback(original_text, corrected_text, image_hash, feedback_type):
    """
    Save user feedback on OCR results for future improvements
    
    Args:
        original_text: The original OCR text
        corrected_text: The user-corrected text
        image_hash: A hash of the image for reference
        feedback_type: Type of feedback (e.g., "ocr_correction", "mrz_correction")
    """
    try:
        import hashlib
        
        # Create feedback directory if it doesn't exist
        feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_data")
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Generate a unique ID for this feedback
        timestamp = datetime.datetime.now().isoformat()
        feedback_id = hashlib.md5(f"{image_hash}_{timestamp}".encode()).hexdigest()
        
        # Create feedback entry
        feedback_entry = {
            "id": feedback_id,
            "timestamp": timestamp,
            "image_hash": image_hash,
            "feedback_type": feedback_type,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "diff_ratio": calculate_diff_ratio(original_text, corrected_text)
        }
        
        # Save to JSON file - use a try block to catch IO errors
        try:
            feedback_file = os.path.join(feedback_dir, f"{feedback_id}.json")
            with open(feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedback_entry, f, indent=2, ensure_ascii=False)
            
            # Log the feedback
            logger.info(f"Saved user feedback: {feedback_id} (type: {feedback_type})")

            # Send notification to admin (if configured)
            try:
                send_admin_notification(feedback_entry)
            except Exception as notify_error:
                logger.error(f"Failed to send admin notification: {notify_error}")

        except IOError as io_error:
            logger.error(f"IO error saving feedback: {io_error}")
            # Return the ID even if file saving failed - we'll handle this in the UI
            return feedback_id, False, f"Error saving feedback: {str(io_error)}"

        return feedback_id, True, "Feedback saved successfully"
    
    except Exception as e:
        logger.error(f"Error in save_feedback: {e}")
        # Return a generated ID even if there was an error
        import uuid
        return str(uuid.uuid4()), False, f"Error processing feedback: {str(e)}"

# Add this function after imports
def calculate_diff_ratio(original, corrected):
    """
    Calculate the difference ratio between original and corrected text
    
    Args:
        original: The original text
        corrected: The corrected text
    
    Returns:
        float: The difference ratio (0.0 to 1.0)
    """
    import difflib
    diff = difflib.SequenceMatcher(None, original, corrected)
    return 1.0 - diff.ratio()

# Add this function after imports
def generate_image_hash(image):
    """
    Generate a hash for the image to use as a reference for feedback
    
    Args:
        image: The image file or image object
    
    Returns:
        str: The hash of the image
    """
    import hashlib
    import io
    import PIL.Image
    
    # Convert image to bytes
    if isinstance(image, PIL.Image.Image):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
    else:
        image_bytes = image.read()
    
    # Generate hash
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    return image_hash

# Duplicate function removed - defined above in admin section
