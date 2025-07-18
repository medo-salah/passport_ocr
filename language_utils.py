import streamlit as st
from config import LANGUAGE_MAP

# UI translations for different languages
UI_TRANSLATIONS = {
    "en": {  # English
        "title": "📷 Passport OCR + Blur Detection + MRZ Extraction",
        "subtitle": "Upload a passport image to extract key info from MRZ and upper fields.",
        "upload_prompt": "Upload Passport Image",
        "drag_drop": "Drag and drop your passport image here",
        "or_browse": "Or click to browse files",
        "processing": "Processing",
        "blur_score": "Blur Score",
        "blur_warning": "Image is too blurry! Please upload a clearer one.",
        "clear_image": "Image is clear. Running OCR...",
        "extracted_text": "Extracted Text",
        "mrz_text": "MRZ OCR Text",
        "extracted_info": "Extracted Info",
        "download_excel": "Download Excel",
        "download_pdf": "Download PDF",
        "no_text": "No text detected",
        "no_mrz": "No MRZ text detected",
        "preprocessing": "Preprocessing Options",
        "rotate": "Rotate Image 180°",
        "denoise": "Denoise Image",
        "grayscale": "Convert to Grayscale",
        "enhance": "Enhance Contrast",
        "language": "Select OCR Language for General Text",
        "theme": "Dark Mode",
        "gpu": "Use GPU (if available)"
    },
    "fr": {  # French
        "title": "📷 OCR de Passeport + Détection de Flou + Extraction MRZ",
        "subtitle": "Téléchargez une image de passeport pour extraire les informations clés de la MRZ et des champs supérieurs.",
        "upload_prompt": "Télécharger l'image du passeport",
        "drag_drop": "Glissez et déposez votre image de passeport ici",
        "or_browse": "Ou cliquez pour parcourir les fichiers",
        "processing": "Traitement",
        "blur_score": "Score de flou",
        "blur_warning": "L'image est trop floue ! Veuillez télécharger une image plus claire.",
        "clear_image": "L'image est claire. Exécution de l'OCR...",
        "extracted_text": "Texte extrait",
        "mrz_text": "Texte OCR MRZ",
        "extracted_info": "Informations extraites",
        "download_excel": "Télécharger Excel",
        "download_pdf": "Télécharger PDF",
        "no_text": "Aucun texte détecté",
        "no_mrz": "Aucun texte MRZ détecté",
        "preprocessing": "Options de prétraitement",
        "rotate": "Rotation de l'image à 180°",
        "denoise": "Débruiter l'image",
        "grayscale": "Convertir en niveaux de gris",
        "enhance": "Améliorer le contraste",
        "language": "Sélectionnez la langue OCR pour le texte général",
        "theme": "Mode sombre",
        "gpu": "Utiliser le GPU (si disponible)"
    },
    "ar": {  # Arabic
        "title": "📷 التعرف الضوئي على جواز السفر + كشف الضبابية + استخراج MRZ",
        "subtitle": "قم بتحميل صورة جواز السفر لاستخراج المعلومات الرئيسية من MRZ والحقول العلوية.",
        "upload_prompt": "تحميل صورة جواز السفر",
        "drag_drop": "اسحب وأفلت صورة جواز السفر هنا",
        "or_browse": "أو انقر لتصفح الملفات",
        "processing": "جاري المعالجة",
        "blur_score": "درجة الضبابية",
        "blur_warning": "الصورة ضبابية جدًا! يرجى تحميل صورة أوضح.",
        "clear_image": "الصورة واضحة. جاري تشغيل التعرف الضوئي...",
        "extracted_text": "النص المستخرج",
        "mrz_text": "نص MRZ المستخرج",
        "extracted_info": "المعلومات المستخرجة",
        "download_excel": "تنزيل إكسل",
        "download_pdf": "تنزيل PDF",
        "no_text": "لم يتم اكتشاف نص",
        "no_mrz": "لم يتم اكتشاف نص MRZ",
        "preprocessing": "خيارات المعالجة المسبقة",
        "rotate": "تدوير الصورة 180 درجة",
        "denoise": "إزالة الضوضاء من الصورة",
        "grayscale": "تحويل إلى تدرج رمادي",
        "enhance": "تحسين التباين",
        "language": "حدد لغة التعرف الضوئي للنص العام",
        "theme": "الوضع الداكن",
        "gpu": "استخدام وحدة معالجة الرسومات (إذا كانت متوفرة)"
    }
}

def get_ui_text(key, lang="en"):
    """Get UI text in the selected language"""
    if lang not in UI_TRANSLATIONS:
        lang = "en"  # Fallback to English
    
    return UI_TRANSLATIONS[lang].get(key, UI_TRANSLATIONS["en"].get(key, key))

def initialize_language():
    """Initialize language settings in session state"""
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"

def set_language(lang):
    """Set the UI language"""
    if lang in UI_TRANSLATIONS:
        st.session_state.ui_language = lang