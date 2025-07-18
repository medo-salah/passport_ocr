import streamlit as st

def initialize_theme():
    """Initialize theme settings in session state"""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"  # Set dark as default

def apply_theme():
    """Apply the dark theme to the app"""
    initialize_theme()
    
    # Common styles
    common_styles = """
    <style>
    /* Make all form labels bold for better visibility */
    label, .stSelectbox label, .stCheckbox label, .stFileUploader label {
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    /* Fix for dropdown options and interactive elements */
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox div[data-baseweb="select"] div,
    .stSelectbox div[data-baseweb="popover"] div,
    .stSelectbox div[data-baseweb="popover"] span,
    .stSelectbox div[data-baseweb="popover"] li,
    .stSelectbox div[data-baseweb="popover"] ul {
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    .stCheckbox, .stRadio, .stSlider, .stFileUploader {
        font-weight: 500 !important;
    }
    </style>
    """
    
    # Dark theme
    dark_theme = """
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: #1e2130;
        color: #fafafa;
    }
    .stTextInput, .stSelectbox {
        background-color: #262730;
        color: #fafafa;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 15px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .upload-prompt {
        color: #4CAF50;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .stTextArea textarea {
        color: #fafafa;
        background-color: #262730;
    }
    .stSelectbox label, .stCheckbox label {
        color: #fafafa !important;
        font-weight: 500 !important;
    }
    .css-1aehpvj, .css-1y4p8pa, .css-16idsys p, .css-183lzff,
    .css-81oif8, .css-10trblm, .css-1vbkxwb, .css-1ht1j8u, .css-1aumxhk {
        color: #fafafa !important;
    }
    .css-16idsys p {
        color: #fafafa !important;
    }
    .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, h6, p, span, div {
        color: #fafafa;
    }
    .upload-header {
        color: #4CAF50;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #fafafa !important;
    }
    div[data-baseweb="popover"] div, 
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        color: #fafafa !important;
        background-color: #262730 !important;
    }
    label, .stSelectbox label, .stRadio label, .stCheckbox label, .stFileUploader label {
        color: #fafafa !important;
        font-weight: 500 !important;
    }
    </style>
    """
    
    # Apply common styles and the dark theme
    st.markdown(common_styles, unsafe_allow_html=True)
    st.markdown(dark_theme, unsafe_allow_html=True)
