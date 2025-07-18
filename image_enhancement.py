import cv2
import numpy as np
from PIL import Image
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, MemoryError) as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"TensorFlow not available: {e}. Auto-rotation detection will use a simpler method.")

def manual_crop_tool(image, key="default"):
    """
    Provide a manual cropping tool for the image
    
    Args:
        image: PIL Image to crop
        key: Unique key for the widget
        
    Returns:
        cropped_image: Cropped PIL Image
    """
    # Get image dimensions
    width, height = image.size
    
    # Create sliders for crop boundaries with unique keys
    st.write("Adjust crop boundaries:")
    
    left = st.slider(
        "Left", 
        0, width//2, 
        0, 
        key=f"crop_left_{key}"
    )
    
    right = st.slider(
        "Right", 
        width//2, width, 
        width, 
        key=f"crop_right_{key}"
    )
    
    top = st.slider(
        "Top", 
        0, height//2, 
        0, 
        key=f"crop_top_{key}"
    )
    
    bottom = st.slider(
        "Bottom", 
        height//2, height, 
        height, 
        key=f"crop_bottom_{key}"
    )
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Show preview
    st.image(cropped_image, caption="Cropped Preview", use_column_width=True)
    
    # Apply crop button
    if st.button("Apply Crop", key=f"apply_crop_btn_{key}"):
        return cropped_image
    
    # Return original if not applied
    return image

def detect_rotation(image):
    """
    Detect if image needs rotation and correct it
    
    Args:
        image: PIL Image to check
        
    Returns:
        rotated_image: Corrected PIL Image
        rotation_angle: Detected rotation angle
    """
    # Check if rotation is disabled in session state
    if 'disable_auto_rotation' in st.session_state and st.session_state.disable_auto_rotation:
        return image, 0
        
    if not TENSORFLOW_AVAILABLE:
        # Use a simpler method without TensorFlow
        st.warning("⚠️ TensorFlow not available. Using basic rotation detection.")
        return detect_rotation_simple(image)
    else:
        try:
            return detect_rotation_with_tf(image)
        except Exception as e:
            logger.error(f"TensorFlow rotation detection failed: {e}")
            st.warning("⚠️ TensorFlow rotation detection failed. Using basic rotation detection.")
            return detect_rotation_simple(image)

def detect_rotation_simple(image):
    """
    Simple rotation detection without TensorFlow
    Uses edge detection and line analysis
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Try all 4 orientations and find the one with most horizontal/vertical lines
    orientations = [0, 90, 180, 270]
    best_score = -1
    best_angle = 0
    
    for angle in orientations:
        # Rotate image
        if angle == 0:
            rotated = gray
        else:
            rotated = np.rot90(gray, k=angle//90)
        
        # Detect edges
        edges = cv2.Canny(rotated, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            continue
        
        # Count horizontal and vertical lines
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):
                h_lines += 1
            else:
                v_lines += 1
        
        # Score based on number of horizontal and vertical lines
        score = h_lines + v_lines
        
        if score > best_score:
            best_score = score
            best_angle = angle
    
    # Only rotate if the score is significantly better than the original orientation
    # This makes the algorithm more conservative
    if best_angle == 0 or best_score < 1.5 * (h_lines + v_lines):
        return image, 0
    else:
        rotated_img = image.rotate(best_angle)
        return rotated_img, best_angle

# This function will only be used if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    # Load rotation detection model (using TF Hub)
    @st.cache_resource
    def load_rotation_model():
        try:
            # Use a small MobileNet model for rotation detection
            model = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=True,
                weights='imagenet',
                pooling='avg'
            )
            return model
        except Exception as e:
            logger.error(f"Error loading rotation model: {e}")
            return None

    def detect_rotation_with_tf(image):
        """
        Detect if image needs rotation using TensorFlow
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Ensure image is RGB (3 channels)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif img_array.shape[2] != 3:  # Not RGB
            st.error("Unsupported image format")
            return image, 0
        
        # Try all 4 orientations and find the one with highest confidence
        orientations = [0, 90, 180, 270]
        best_score = -1
        best_angle = 0
        original_score = -1
        
        try:
            model = load_rotation_model()
            if model is None:
                return image, 0
            
            for angle in orientations:
                # Rotate image
                if angle == 0:
                    rotated = img_array
                else:
                    rotated = np.rot90(img_array, k=angle//90)
                
                # Resize for model input
                resized = cv2.resize(rotated, (224, 224))
                
                # Ensure image is RGB (3 channels)
                if resized.shape[2] != 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
                
                # Preprocess for model
                preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
                preprocessed = np.expand_dims(preprocessed, axis=0)
                
                try:
                    # Get model prediction
                    predictions = model.predict(preprocessed, verbose=0)
                    
                    # Use top-5 accuracy as a confidence score
                    top5 = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]
                    score = sum(score for _, _, score in top5)
                    
                    if angle == 0:
                        original_score = score
                    
                    if score > best_score:
                        best_score = score
                        best_angle = angle
                except Exception as e:
                    logger.error(f"Error during prediction for angle {angle}: {e}")
                    continue
            
            # Only rotate if the best score is significantly better than the original orientation
            # This makes the algorithm more conservative
            if best_angle == 0 or best_score < 1.2 * original_score:
                return image, 0
            else:
                rotated_img = image.rotate(best_angle)
                return rotated_img, best_angle
                
        except Exception as e:
            logger.error(f"TensorFlow rotation detection error: {e}")
            # Fall back to simple rotation detection
            return detect_rotation_simple(image)





