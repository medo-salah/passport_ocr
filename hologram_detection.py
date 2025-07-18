import cv2
import numpy as np
from PIL import Image
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def detect_fake_document(image):
    """
    Perform basic authenticity checks to detect potentially fake documents
    
    Args:
        image: PIL Image or numpy array of document
        
    Returns:
        marked_image: Visualization of suspicious areas
        detection_data: Dictionary with detection results
    """
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Create a copy for visualization
    vis_image = img_array.copy()
    
    # Convert to BGR if in RGB format (for OpenCV processing)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Initialize detection results
    suspicious_areas = []
    confidence_score = 0.0
    issues_found = []
    
    # 1. Check for digital manipulation using Error Level Analysis (ELA)
    ela_score = check_digital_manipulation(img_array)
    if ela_score > 0.6:
        issues_found.append("Possible digital manipulation detected")
        confidence_score += 0.3
    
    # 2. Check for inconsistent text (common in fake documents)
    text_issues, text_regions = check_text_inconsistency(img_bgr)
    if text_issues:
        issues_found.extend(text_issues)
        suspicious_areas.extend(text_regions)
        confidence_score += 0.25
    
    # 3. Check for printing artifacts (common in scanned fakes)
    print_issues, print_regions = check_printing_artifacts(img_bgr)
    if print_issues:
        issues_found.extend(print_issues)
        suspicious_areas.extend(print_regions)
        confidence_score += 0.25
    
    # 4. Check for color inconsistencies
    color_issues, color_regions = check_color_consistency(img_bgr)
    if color_issues:
        issues_found.extend(color_issues)
        suspicious_areas.extend(color_regions)
        confidence_score += 0.2
    
    # Draw suspicious areas on visualization image
    for area in suspicious_areas:
        x, y, w, h = area
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(vis_image, "Suspicious", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Prepare result data
    detection_data = {
        "fake_detected": len(issues_found) > 0,
        "confidence": min(0.95, confidence_score),
        "issues": issues_found,
        "suspicious_areas": suspicious_areas
    }
    
    # Convert back to PIL for return
    return Image.fromarray(vis_image), detection_data

def check_digital_manipulation(image):
    """
    Check for digital manipulation using Error Level Analysis
    
    Args:
        image: Numpy array of image
        
    Returns:
        score: Manipulation likelihood score (0-1)
    """
    # Simple implementation of Error Level Analysis
    # In a real implementation, this would save and reload the image at different qualities
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Apply Laplacian filter to detect inconsistencies
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate statistics
    edge_mean = np.mean(edges)
    laplacian_std = np.std(laplacian)
    
    # Higher values indicate potential manipulation
    # This is a simplified approach - real ELA is more complex
    score = min(1.0, max(0.0, (edge_mean / 255.0) * (laplacian_std / 50.0)))
    
    return score

def check_text_inconsistency(image):
    """
    Check for inconsistent text (font, size, alignment)
    
    Args:
        image: BGR image
        
    Returns:
        issues: List of detected issues
        regions: List of suspicious regions (x, y, w, h)
    """
    issues = []
    regions = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get text regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find text contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for text-like contours
    text_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        
        # Text typically has certain aspect ratios and sizes
        if 0.1 < aspect_ratio < 15 and area > 100 and area < 10000:
            text_contours.append((x, y, w, h))
    
    # Check for inconsistent text heights (potential sign of tampering)
    if len(text_contours) > 5:  # Need enough text to analyze
        heights = [h for _, _, _, h in text_contours]
        mean_height = np.mean(heights)
        std_height = np.std(heights)
        
        # Flag highly variable text heights
        if std_height / mean_height > 0.5:
            issues.append("Inconsistent text heights detected")
            
            # Find outlier text regions
            for x, y, w, h in text_contours:
                if abs(h - mean_height) > 2 * std_height:
                    regions.append((x, y, w, h))
    
    return issues, regions

def check_printing_artifacts(image):
    """
    Check for printing and scanning artifacts
    
    Args:
        image: BGR image
        
    Returns:
        issues: List of detected issues
        regions: List of suspicious regions (x, y, w, h)
    """
    issues = []
    regions = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check for moiré patterns (common in scanned documents)
    # Apply FFT to detect regular patterns
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Check for strong peaks in frequency domain (indicates regular patterns)
    threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
    strong_frequencies = np.sum(magnitude_spectrum > threshold)
    
    if strong_frequencies > 100:
        issues.append("Possible scan artifacts detected")
        # Mark the whole document as suspicious in this case
        h, w = gray.shape
        regions.append((0, 0, w, h))
    
    # Check for printer dot patterns
    # Apply bandpass filter to isolate printer dot frequencies
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq = cv2.subtract(gray, blur)
    
    # Check for regular patterns in high frequency components
    _, high_thresh = cv2.threshold(high_freq, 15, 255, cv2.THRESH_BINARY)
    
    # Count isolated dots
    dot_count = np.sum(high_thresh > 0)
    if dot_count > gray.size * 0.01:  # More than 1% of pixels are dots
        issues.append("Printer dot patterns detected")
    
    return issues, regions

def check_color_consistency(image):
    """
    Check for color inconsistencies that might indicate tampering
    
    Args:
        image: BGR image
        
    Returns:
        issues: List of detected issues
        regions: List of suspicious regions (x, y, w, h)
    """
    issues = []
    regions = []
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split channels
    h, s, v = cv2.split(hsv)
    
    # Check for abrupt color transitions (potential signs of editing)
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(h, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Find areas with abrupt color changes
    threshold = np.mean(gradient_magnitude) + 2 * np.std(gradient_magnitude)
    _, binary = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    
    # Find contours of suspicious areas
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for significant areas
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum size to be considered
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
    
    if regions:
        issues.append("Suspicious color transitions detected")
    
    return issues, regions

def verify_security_features(image):
    """
    Verify document authenticity and assess security features
    
    Args:
        image: PIL Image of document
        
    Returns:
        result: Dictionary with verification results
        score: Overall authenticity score (0-1)
    """
    # Run fake document detection
    _, detection_data = detect_fake_document(image)
    
    # Calculate authenticity score (inverse of fake detection confidence)
    authenticity_score = 1.0 - detection_data["confidence"]
    
    # Prepare security features assessment
    security_features = {
        "microprint": "Not analyzed",
        "optically_variable": "Not analyzed",
        "kinegram": "Not analyzed",
        "uv_reactive": "Not analyzed",
        "metallic_ink": "Not analyzed"
    }
    
    # Get assessment text
    if authenticity_score > 0.8:
        assessment = "Document appears authentic"
    elif authenticity_score > 0.5:
        assessment = "Document has some suspicious elements"
    else:
        assessment = "Document has multiple suspicious elements"
    
    # Prepare result
    result = {
        "authenticity_score": authenticity_score,
        "issues_detected": detection_data["issues"],
        "security_features": security_features,
        "assessment": assessment
    }
    
    return result, authenticity_score

# Keep this function for backward compatibility
def detect_holograms(image):
    """
    Redirects to fake document detection for backward compatibility
    """
    return detect_fake_document(image)

