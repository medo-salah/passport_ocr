#!/usr/bin/env python3
"""
GPU Diagnostic and Fix Script for PyTorch CUDA Issues
Specifically designed to help with NVIDIA Driver 576.52 compatibility issues
"""

import subprocess
import sys
import os
import platform

def run_command(command, description=""):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description or command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return False, "", "Timeout"
    except Exception as e:
        print(f"Error running command: {e}")
        return False, "", str(e)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("\n🔍 Checking NVIDIA Driver...")
    
    success, stdout, stderr = run_command("nvidia-smi", "Check NVIDIA driver")
    
    if success:
        print("✅ NVIDIA driver is installed and working")
        # Extract driver version
        lines = stdout.split('\n')
        for line in lines:
            if "Driver Version:" in line:
                driver_version = line.split("Driver Version:")[1].split()[0]
                print(f"📋 Driver Version: {driver_version}")
                return True, driver_version
    else:
        print("❌ NVIDIA driver not found or not working")
        return False, None
    
    return True, "Unknown"

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    print("\n🔍 Checking CUDA Toolkit...")
    
    success, stdout, stderr = run_command("nvcc --version", "Check CUDA toolkit")
    
    if success:
        print("✅ CUDA toolkit is installed")
        # Extract CUDA version
        lines = stdout.split('\n')
        for line in lines:
            if "release" in line.lower():
                cuda_version = line.split("release")[1].split(",")[0].strip()
                print(f"📋 CUDA Version: {cuda_version}")
                return True, cuda_version
    else:
        print("❌ CUDA toolkit not found")
        print("💡 You may need to install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads")
        return False, None
    
    return True, "Unknown"

def check_pytorch():
    """Check current PyTorch installation"""
    print("\n🔍 Checking PyTorch Installation...")
    
    try:
        import torch
        print("✅ PyTorch is installed")
        print(f"📋 PyTorch Version: {torch.__version__}")
        print(f"📋 CUDA Available: {torch.cuda.is_available()}")
        print(f"📋 CUDA Version (PyTorch): {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"📋 GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"📋 GPU {i}: {torch.cuda.get_device_name(i)}")
            return True, torch.__version__, torch.version.cuda
        else:
            print("❌ PyTorch cannot access CUDA")
            return False, torch.__version__, torch.version.cuda
    
    except ImportError:
        print("❌ PyTorch is not installed")
        return False, None, None
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False, None, None

def fix_pytorch_cuda():
    """Fix PyTorch CUDA installation"""
    print("\n🔧 Fixing PyTorch CUDA Installation...")
    
    # Uninstall current PyTorch
    print("📤 Uninstalling current PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", "Uninstall PyTorch")
    
    # Install PyTorch with CUDA support
    print("📥 Installing PyTorch with CUDA 11.8 support...")
    success, stdout, stderr = run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "Install PyTorch with CUDA 11.8"
    )
    
    if not success:
        print("⚠️ CUDA 11.8 installation failed, trying CUDA 12.1...")
        success, stdout, stderr = run_command(
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "Install PyTorch with CUDA 12.1"
        )
    
    if success:
        print("✅ PyTorch installation completed")
        return True
    else:
        print("❌ PyTorch installation failed")
        return False

def verify_installation():
    """Verify the fixed installation"""
    print("\n✅ Verifying Installation...")
    
    success, stdout, stderr = run_command(
        'python -c "import torch; print(f\'PyTorch: {torch.__version__}\'); print(f\'CUDA Available: {torch.cuda.is_available()}\'); print(f\'CUDA Version: {torch.version.cuda}\')"',
        "Verify PyTorch CUDA"
    )
    
    if success and "CUDA Available: True" in stdout:
        print("🎉 SUCCESS! PyTorch can now access CUDA!")
        return True
    else:
        print("❌ Verification failed - PyTorch still cannot access CUDA")
        return False

def main():
    """Main diagnostic and fix function"""
    print("🚀 GPU Diagnostic and Fix Script")
    print("=" * 60)
    print("This script will help fix PyTorch CUDA issues with NVIDIA Driver 576.52")
    print("=" * 60)
    
    # Check system info
    print(f"🖥️  Operating System: {platform.system()} {platform.release()}")
    print(f"🐍 Python Version: {sys.version}")
    
    # Step 1: Check NVIDIA driver
    driver_ok, driver_version = check_nvidia_driver()
    
    if not driver_ok:
        print("\n❌ NVIDIA driver issue detected!")
        print("💡 Please install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return False
    
    # Step 2: Check CUDA toolkit
    cuda_ok, cuda_version = check_cuda_toolkit()
    
    # Step 3: Check PyTorch
    pytorch_ok, pytorch_version, pytorch_cuda = check_pytorch()
    
    # Analyze the situation
    print("\n📊 ANALYSIS:")
    print(f"✅ NVIDIA Driver: {driver_version}")
    print(f"{'✅' if cuda_ok else '❌'} CUDA Toolkit: {cuda_version or 'Not found'}")
    print(f"{'✅' if pytorch_ok else '❌'} PyTorch CUDA: {pytorch_cuda or 'Not available'}")
    
    if pytorch_ok:
        print("\n🎉 Everything looks good! PyTorch can access CUDA.")
        return True
    
    # Offer to fix the issue
    print("\n🔧 RECOMMENDED ACTIONS:")
    
    if not cuda_ok:
        print("1. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("2. Restart your computer")
        print("3. Run this script again")
        return False
    
    print("1. Reinstall PyTorch with proper CUDA support")
    print("2. The script can do this automatically")
    
    response = input("\n❓ Would you like to automatically fix PyTorch? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        if fix_pytorch_cuda():
            return verify_installation()
        else:
            print("\n❌ Automatic fix failed. Please try manual installation:")
            print("pip uninstall torch torchvision torchaudio")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    else:
        print("\n💡 Manual fix instructions:")
        print("1. pip uninstall torch torchvision torchaudio")
        print("2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Restart your application")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 All done! Your GPU should now work with PyTorch.")
        else:
            print("\n⚠️  Some issues remain. Please follow the manual instructions above.")
    except KeyboardInterrupt:
        print("\n\n⏹️  Script interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue with the full error message.")
