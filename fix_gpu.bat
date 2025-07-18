@echo off
echo ========================================
echo GPU Diagnostic and Fix Script
echo ========================================
echo.
echo This script will help fix PyTorch CUDA issues
echo with NVIDIA Driver 576.52
echo.
echo Press any key to continue...
pause >nul

python fix_gpu.py

echo.
echo ========================================
echo Script completed!
echo ========================================
echo.
echo If the issue persists, please:
echo 1. Restart your computer
echo 2. Run this script again
echo 3. Check the Streamlit app
echo.
pause
