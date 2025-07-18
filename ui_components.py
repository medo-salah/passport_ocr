import streamlit as st
import time
from contextlib import contextmanager

@contextmanager
def progress_bar(label="Processing"):
    """Context manager that displays a progress bar during execution"""
    progress = st.progress(0)
    status_text = st.empty()
    
    # Show initial status
    status_text.text(f"{label}... 0%")
    
    try:
        # Start progress tracking
        for i in range(1, 101):
            # Yield control back to the caller at 1%
            if i == 1:
                yield
                
            # Update progress bar
            progress.progress(i)
            status_text.text(f"{label}... {i}%")
            
            # Slow down progress updates for visual effect
            time.sleep(0.01)
    finally:
        # Ensure we show 100% at the end
        progress.progress(100)
        status_text.text(f"{label} complete!")
        time.sleep(0.5)
        
        # Clean up
        progress.empty()
        status_text.empty()