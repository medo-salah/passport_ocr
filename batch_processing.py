import streamlit as st
import pandas as pd
import io
import zipfile
from PIL import Image
import numpy as np
import cv2
import logging
from utils.export_utils import export_to_excel, export_to_pdf

logger = logging.getLogger(__name__)

def process_batch(files, process_function, progress_callback=None):
    """
    Process multiple passport images in batch
    
    Args:
        files: List of uploaded files
        process_function: Function to process each image
        progress_callback: Optional callback to update progress
        
    Returns:
        results: List of processing results
        failed: List of failed files
    """
    results = []
    failed = []
    
    total_files = len(files)
    
    for i, file in enumerate(files):
        try:
            # Update progress if callback provided
            if progress_callback:
                progress_callback((i + 1) / total_files, f"Processing {file.name} ({i+1}/{total_files})")
            
            # Process the file
            result = process_function(file)
            results.append({
                "filename": file.name,
                "data": result
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            failed.append({
                "filename": file.name,
                "error": str(e)
            })
    
    return results, failed

def export_batch_results(results):
    """
    Export batch processing results to Excel and PDF
    
    Args:
        results: List of processing results
        
    Returns:
        excel_bytes: Excel file as bytes
        pdf_bytes: PDF file as bytes
        zip_bytes: ZIP file containing individual PDFs
    """
    # Create Excel with all results
    excel_buffer = io.BytesIO()
    
    # Extract data from results
    all_data = []
    for result in results:
        data = result["data"].copy()
        data["Filename"] = result["filename"]
        all_data.append(data)
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(all_data)
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    # Create ZIP file with individual PDFs
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for result in results:
            pdf_bytes = export_to_pdf(result["data"])
            filename = result["filename"].split('.')[0] + '.pdf'
            zip_file.writestr(filename, pdf_bytes)
    
    # Create a summary PDF
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Batch Processing Results", ln=True, align='C')
    pdf.ln(10)
    
    for i, result in enumerate(results):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt=f"Document {i+1}: {result['filename']}", ln=True)
        pdf.set_font("Arial", size=10)
        
        for key, value in result["data"].items():
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
        
        pdf.ln(5)
    
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    
    return excel_buffer.getvalue(), pdf_buffer.getvalue(), zip_buffer.getvalue()

def process_single_image(file, preprocessing_options, ocr_lang, use_gpu):
    """Process a single image file and extract data with validation"""
    try:
        # Open the image
        image = Image.open(file)
        
        # Apply preprocessing
        from image_processing import preprocess_image
        processed_image = preprocess_image(
            image,
            denoise=preprocessing_options["denoise"],
            grayscale=preprocessing_options["grayscale"],
            enhance_contrast=preprocessing_options["enhance"]
        )
        
        # Convert to OpenCV format
        import cv2
        import numpy as np
        image_cv = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if image_cv.shape[0] > 1200:
            scale = 1200 / image_cv.shape[0]
            image_cv = cv2.resize(image_cv, None, fx=scale, fy=scale)
        
        # Check if image is blurry
        from utils.blur_detection import is_blurry
        blurry, score = is_blurry(processed_image)
        if blurry:
            return {"error": "Image too blurry", "blur_score": score}
        
        # OCR processing
        try:
            import easyocr
            from config import LANGUAGE_MAP
            reader = easyocr.Reader([LANGUAGE_MAP[ocr_lang]], gpu=use_gpu)
            results = reader.readtext(np.array(processed_image), detail=1)
            text = "\n".join([r[1] for r in results])
        except Exception as e:
            return {"error": f"OCR error: {str(e)}"}
        
        # MRZ processing
        from mrz_processing import extract_mrz_fields, detect_mrz_region
        mrz_image, mrz_text = detect_mrz_region(image_cv)
        
        # Data extraction with validation
        from ner_processing import load_spacy_model, extract_with_ner
        import streamlit as st
        nlp = load_spacy_model()
        data = extract_mrz_fields(mrz_text)
        data.update(extract_with_ner(text, nlp))
        
        # Barcode extraction
        from utils.barcode_reader import extract_barcodes
        barcodes = extract_barcodes(image_cv)
        if barcodes:
            data["Barcodes"] = ", ".join(barcodes)
        
        # Check validation results
        if "_validation" in data:
            # Count validation issues
            validation_issues = sum(1 for result in data["_validation"].values() if not result["valid"])
            data["Validation Issues"] = validation_issues
            
            # Add validation summary
            critical_fields = ["Passport Number", "Date of Birth", "Expiration Date"]
            invalid_critical = [field for field in critical_fields 
                               if field in data["_validation"] and not data["_validation"][field]["valid"]]
            
            if invalid_critical:
                data["Critical Issues"] = ", ".join(invalid_critical)
    
    except Exception as e:
        error_type = "image_processing_error"
        if "tesseract" in str(e).lower():
            error_type = "tesseract_not_found"
        elif "cuda" in str(e).lower() or "gpu" in str(e).lower():
            error_type = "gpu_error"
        elif "ocr" in str(e).lower():
            error_type = "ocr_processing_error"
        elif "spacy" in str(e).lower() or "nlp" in str(e).lower():
            error_type = "spacy_model_error"
        elif "mrz" in str(e).lower():
            error_type = "mrz_detection_failed"
            
        error_details = get_error_explanation(error_type, str(e))
        return {
            "error": f"{str(e)}",
            "error_type": error_type,
            "explanation": error_details["explanation"],
            "suggested_fix": error_details["suggested_fixes"][0]
        }

    return data



