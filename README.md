# 📷 Passport OCR & Document Verification System

A world-class, high-performance solution for extracting data from passports and ID cards. This system combines traditional Computer Vision (OpenCV), Deep Learning OCR (EasyOCR), and Natural Language Processing (spaCy) to provide accurate data extraction, authenticity verification, and document analysis.

## 🚀 Key Features

*   **Advanced OCR Extraction**: Extracts text using EasyOCR with GPU acceleration support.
*   **MRZ Processing**: Automatically detects and parses the Machine Readable Zone (MRZ) according to ICAO standards.
*   **Document Authenticity**: Checks for digital manipulation (Error Level Analysis), printing artifacts, and inconsistent text.
*   **Facial Analysis**: Detects faces in documents and verifies if they meet quality requirements (brightness, contrast, resolution).
*   **Batch Processing**: Support for uploading and processing multiple documents simultaneously with Excel/PDF export.
*   **FastAPI Integration**: A production-ready REST API for programmatic document processing.
*   **Admin Dashboard**: Real-time analytics, database management (SQLite), and system health monitoring.
*   **Privacy & GDPR**: In-memory processing options, automatic data deletion timers, and sensitive data masking.

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Backend API**: [FastAPI](https://fastapi.tiagolo.org/)
*   **OCR Engines**: [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
*   **Computer Vision**: [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
*   **NLP**: [spaCy](https://spacy.io/)
*   **Database**: SQLite

## 📋 Prerequisites

1.  **Python 3.9+**
2.  **Tesseract OCR**: Install Tesseract on your system.
    *   Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
    *   Set the environment variable `TESSDATA_PREFIX` to your `tessdata` folder.
3.  **NVIDIA GPU (Optional)**: Highly recommended for faster OCR. Ensure CUDA and cuDNN are configured.

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/passport-ocr.git
    cd passport-ocr
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3.  **(Optional) Fix GPU Issues**:
    If you have an NVIDIA GPU but PyTorch isn't detecting it, run the included diagnostic script:
    ```bash
    python fix_gpu.py
    ```

## 🚦 Usage

### Running the Streamlit App
The main user interface for document processing and administration.
```bash
streamlit run main.py
```

### Running the API Server
For integrating document processing into your own applications.
```bash
python api.py
```
Once running, visit `http://localhost:8000/docs` to view the interactive Swagger API documentation.

## 📂 Project Structure

*   `main.py`: Entry point for the Streamlit application.
*   `api.py`: FastAPI implementation.
*   `mrz_processing.py`: Logic for detecting and parsing MRZ zones.
*   `hologram_detection.py`: Security and authenticity verification logic.
*   `document_analysis.py`: Face detection and document classification.
*   `database_manager.py`: Handles SQLite storage for OCR results and analytics.
*   `analytics_dashboard.py`: Visualizations for system usage.

## 🛡️ Admin Access
The application includes an Admin tab for system monitoring and data management. 
*   **Default Password**: `admin123` (Change this in the Admin Settings tab upon first login).

## 🔒 Privacy & Security
This application is built with privacy in mind:
*   **Client-Side Processing**: Option to run OCR locally via Tesseract.
*   **Data Masking**: Hide sensitive fields like Passport Numbers in the UI.
*   **Auto-Deletion**: Configurable timer to wipe session data from memory.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Disclaimer: This tool is intended for data extraction assistance. Always verify the results against the physical document.*