#!/usr/bin/env python
import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for the Passport OCR API"""
    print("Installing required Python packages...")
    
    # Install packages from requirements file
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-api.txt"])
    
    # Install spaCy model
    print("Installing spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("All dependencies installed successfully!")
    print("You can now run the API with: python api.py")

if __name__ == "__main__":
    install_dependencies()