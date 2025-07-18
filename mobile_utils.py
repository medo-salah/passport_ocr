import streamlit as st

def detect_mobile():
    """Detect if user is on a mobile device based on viewport width"""
    mobile_css = """
    <style>
    .mobile-container {
        display: none;
    }
    
    @media (max-width: 640px) {
        .desktop-container {
            display: none !important;
        }
        .mobile-container {
            display: block !important;
        }
        .stButton button {
            width: 100%;
        }
        .stTextInput input, .stSelectbox, .stFileUploader {
            width: 100% !important;
        }
    }
    </style>
    """
    
    st.markdown(mobile_css, unsafe_allow_html=True)
    
    # Add JavaScript to detect viewport width and set a session variable
    st.markdown("""
    <script>
        // Set a flag in localStorage based on viewport width
        if (window.innerWidth <= 640) {
            localStorage.setItem('is_mobile', 'true');
        } else {
            localStorage.setItem('is_mobile', 'false');
        }
    </script>
    """, unsafe_allow_html=True)

def mobile_layout_wrapper(content_function):
    """Wrapper to apply mobile-specific layout adjustments"""
    detect_mobile()
    
    # Desktop container
    st.markdown('<div class="desktop-container">', unsafe_allow_html=True)
    content_function(is_mobile=False)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mobile container
    st.markdown('<div class="mobile-container">', unsafe_allow_html=True)
    content_function(is_mobile=True)
    st.markdown('</div>', unsafe_allow_html=True)