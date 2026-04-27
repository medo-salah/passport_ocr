import os
import sys
import subprocess
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Runs a system command and logs the process."""
    logger.info(f"Starting: {description}")
    try:
        subprocess.check_call(command, shell=True)
        logger.info(f"Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}. Error: {e}")
        return False

def install_tesseract():
    """Attempts to install Tesseract OCR based on the operating system."""
    os_type = platform.system()
    logger.info(f"Detected OS: {os_type}")

    if os_type == "Windows":
        # Check if already installed in default locations
        common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        if any(os.path.exists(path) for path in common_paths):
            logger.info("Tesseract OCR is already installed on Windows.")
            return True
        
        logger.info("Tesseract not found. Attempting installation via winget...")
        # winget is available on Windows 10 (1709+) and Windows 11
        success = run_command("winget install -e --id UB-Mannheim.TesseractOCR", "Tesseract OCR via winget")
        if not success:
            logger.warning("Automated Windows install failed. Please download manually from: https://github.com/UB-Mannheim/tesseract/wiki")
        return success

    elif os_type == "Linux":
        return run_command("sudo apt-get update && sudo apt-get install -y tesseract-ocr", "Tesseract OCR via apt")

    elif os_type == "Darwin":  # macOS
        return run_command("brew install tesseract", "Tesseract OCR via Homebrew")

    else:
        logger.error(f"Unsupported OS for automated Tesseract installation: {os_type}")
        return False

def main():
    """Main setup routine."""
    print("====================================================")
    print("   Passport OCR & Verification System Setup")
    print("====================================================\n")

    # 1. Install Requirements
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Python dependencies")
    else:
        logger.error("requirements.txt not found. Skipping dependency installation.")

    # 2. Install spaCy Model
    run_command(f"{sys.executable} -m spacy download en_core_web_sm", "spaCy English model (en_core_web_sm)")

    # 3. Install Tesseract
    install_tesseract()

    print("\n====================================================")
    print("Setup process finished.")
    print("If Tesseract was just installed, you may need to restart your terminal.")
    print("Run the app: streamlit run main.py")
    print("====================================================")

if __name__ == "__main__":
    main()