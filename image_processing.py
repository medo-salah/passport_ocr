import cv2
import numpy as np
from PIL import Image
import streamlit as st
from language_utils import get_ui_text

def preprocess_image(image, denoise=True, grayscale=True, enhance_contrast=True):
    """Apply preprocessing to improve OCR accuracy"""
    img_array = np.array(image)
    
    if grayscale:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_array = clahe.apply(img_array) if grayscale else clahe.apply(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
    
    if denoise:
        img_array = cv2.fastNlMeansDenoising(img_array, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return Image.fromarray(img_array)

def setup_preprocessing_sidebar(context="default"):
    """
    Setup preprocessing options in the sidebar
    
    Args:
        context: Context identifier to ensure unique widget keys
    """
    ui_lang = getattr(st.session_state, "ui_language", "en")
    st.sidebar.title(f"🛠️ {get_ui_text('preprocessing', ui_lang)}")
    
    options = {
        "rotate": st.sidebar.checkbox(
            get_ui_text("rotate", ui_lang), 
            key=f"preprocess_rotate_{context}_{ui_lang}"
        ),
        "denoise": st.sidebar.checkbox(
            get_ui_text("denoise", ui_lang), 
            key=f"preprocess_denoise_{context}_{ui_lang}"
        ),
        "grayscale": st.sidebar.checkbox(
            get_ui_text("grayscale", ui_lang), 
            key=f"preprocess_grayscale_{context}_{ui_lang}"
        ),
        "enhance": st.sidebar.checkbox(
            get_ui_text("enhance", ui_lang), 
            key=f"preprocess_enhance_{context}_{ui_lang}"
        )
    }
    return options
