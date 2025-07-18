import torch
import subprocess
import streamlit as st
import time

def get_gpu_info():
    """Returns detailed GPU information and troubleshooting tips"""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.version.cuda else "Not available",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "driver_issue": False,
        "troubleshooting_steps": []
    }
    
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        driver_version = result.stdout.strip() if result.returncode == 0 else "Unknown"
        gpu_info["nvidia_driver"] = driver_version
        
        if not gpu_info["cuda_available"]:
            gpu_info["driver_issue"] = True
            gpu_info["troubleshooting_steps"] = [
                "1. **Check PyTorch Installation**: Run `python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"`",
                "2. **Verify CUDA Toolkit**: Run `nvcc --version` in terminal",
                "3. **Current Driver**: " + driver_version + " (Latest drivers recommended)",
                "4. **Reinstall PyTorch with CUDA**: Choose the correct version:",
                "   • For CUDA 11.8: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`",
                "   • For CUDA 12.1: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`",
                "5. **Environment Variables**: Ensure CUDA_PATH and PATH are set correctly",
                "6. **Restart Required**: Restart your system after driver/CUDA updates"
            ]
    except Exception:
        gpu_info["nvidia_driver"] = "Not detected"

    return gpu_info

def display_gpu_status(context="main"):
    """Displays GPU status with a professional troubleshooting guide"""
    gpu_info = get_gpu_info()
    use_gpu = gpu_info["cuda_available"]
    
    if gpu_info["cuda_available"]:
        st.success(f"✅ CUDA GPU Detected: {gpu_info['gpu_name']} (Driver: {gpu_info['nvidia_driver']}, CUDA: {gpu_info['cuda_version']})")
    else:
        if gpu_info["driver_issue"]:
            # Create a container for the warning
            warning_container = st.empty()
            
            # Display warning with icon that will fade out
            warning_container.warning(f"🔧 NVIDIA Driver {gpu_info['nvidia_driver']} detected but PyTorch can't access GPU")
            
            # Create a container for the troubleshooting guide that will remain
            guide_container = st.container()
            
            # Add CSS for fade-out animation
            st.markdown("""
            <style>
            @keyframes fadeOut {
                0% { opacity: 1; }
                100% { opacity: 0; visibility: hidden; }
            }
            .fade-out {
                animation: fadeOut 1.5s ease-in-out 5s forwards;
            }
            </style>
            <div class="fade-out">
                <div id="warning-placeholder"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Use JavaScript to replace the warning after a delay
            st.markdown("""
            <script>
                setTimeout(function() {
                    const warning = document.querySelector('.stAlert');
                    if (warning) {
                        document.getElementById('warning-placeholder').appendChild(warning.cloneNode(true));
                        warning.style.display = 'none';
                    }
                }, 100);
            </script>
            """, unsafe_allow_html=True)
            
            # Create a professional troubleshooting guide that remains visible
            with guide_container.expander("📋 Technical Diagnostics & Resolution Steps", expanded=False):
                st.markdown("### GPU Compatibility Diagnostics")
                
                # Create a two-column layout for the diagnostics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### System Information")
                    st.markdown(f"* **NVIDIA Driver**: {gpu_info['nvidia_driver']}")
                    st.markdown(f"* **CUDA Version**: {gpu_info['cuda_version']}")
                    st.markdown(f"* **GPU Model**: {gpu_info['gpu_name']}")
                
                with col2:
                    st.markdown("#### Resolution Steps")
                    for step in gpu_info["troubleshooting_steps"]:
                        st.markdown(f"{step}")
                
                # Add a divider
                st.divider()
                
                # Add quick diagnostic tests
                st.markdown("#### Quick Diagnostic Tests")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🧪 Test CUDA Installation", key=f"gpu_test_cuda_install_{context}"):
                        try:
                            import torch
                            st.write(f"**PyTorch Version**: {torch.__version__}")
                            st.write(f"**CUDA Available**: {torch.cuda.is_available()}")
                            st.write(f"**CUDA Version**: {torch.version.cuda}")
                            if torch.cuda.is_available():
                                st.write(f"**GPU Count**: {torch.cuda.device_count()}")
                                st.write(f"**Current Device**: {torch.cuda.current_device()}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                with col2:
                    if st.button("🔧 Check NVIDIA-SMI", key=f"gpu_check_nvidia_smi_{context}"):
                        try:
                            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                st.success("✅ NVIDIA-SMI working")
                                with st.expander("NVIDIA-SMI Output"):
                                    st.code(result.stdout)
                            else:
                                st.error("❌ NVIDIA-SMI failed")
                                st.code(result.stderr)
                        except Exception as e:
                            st.error(f"Error running nvidia-smi: {e}")

                # Add installation commands
                st.markdown("#### Installation Commands")

                st.markdown("**Uninstall current PyTorch:**")
                st.code("pip uninstall torch torchvision torchaudio")

                st.markdown("**Install PyTorch with CUDA 11.8:**")
                st.code("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

                st.markdown("**Install PyTorch with CUDA 12.1:**")
                st.code("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

                st.markdown("**Verify installation:**")
                st.code('python -c "import torch; print(f\'CUDA available: {torch.cuda.is_available()}\')"')

                # Add additional resources
                st.markdown("#### Additional Resources")
                st.markdown("* [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)")
                st.markdown("* [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)")
                st.markdown("* [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)")
                st.markdown("* [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/cuda.html)")
        else:
            st.warning("❌ No CUDA-capable GPU detected. Using CPU only")
    
    return use_gpu

def get_gpu_utilization(device=None):
    """Returns the GPU utilization percentage if available"""
    if not torch.cuda.is_available():
        return None
    
    try:
        return torch.cuda.utilization(device)
    except Exception:
        return None

def get_gpu_memory_info(device=None):
    """Returns GPU memory information if available"""
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get current device if not specified
        if device is None:
            device = torch.cuda.current_device()
            
        # Get memory information
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        
        return {
            "allocated_mb": round(memory_allocated, 2),
            "reserved_mb": round(memory_reserved, 2),
            "max_allocated_mb": round(max_memory_allocated, 2)
        }
    except Exception:
        return None

def list_running_processes():
    """Returns information about GPU processes if available"""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    try:
        return torch.cuda.list_gpu_processes()
    except Exception:
        return "Unable to retrieve GPU process information"

def empty_gpu_cache():
    """Empties the GPU cache if CUDA is available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return True
    return False
