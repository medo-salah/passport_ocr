import cv2
import numpy as np
import re
import logging
import pytesseract
from PIL import Image
import streamlit as st

logger = logging.getLogger(__name__)

# Add validation functions that might be missing
def validate_date(date_str):
    """Validate date format YYYY-MM-DD"""
    if not date_str or date_str == "Not Found":
        return False, "Date not found"
    
    try:
        import datetime
        # Check if date is in YYYY-MM-DD format
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True, "Valid date"
    except ValueError:
        return False, "Invalid date format"

def validate_country_code(code):
    """Validate 3-letter country code"""
    if not code or code == "Not Found":
        return False, "Country code not found"
    
    if len(code) == 3 and code.isalpha():
        return True, "Valid country code"
    return False, "Invalid country code format"

def validate_passport_number(number):
    """Validate passport number format"""
    if not number or number == "Not Found":
        return False, "Passport number not found"
    
    if len(number) >= 7 and len(number) <= 9:
        return True, "Valid passport number format"
    return False, "Invalid passport number format"

def validate_sex(sex):
    """Validate sex code (M/F)"""
    if not sex or sex == "Not Found":
        return False, "Sex code not found"
    
    if sex in ["M", "F"]:
        return True, "Valid sex code"
    return False, "Invalid sex code"

def extract_mrz_fields(text):
    data = {
        "Document Type": "Not Found",
        "Issuing Country": "Not Found",
        "Full Name": "Not Found",
        "Date of Birth": "Not Found",
        "Sex": "Not Found",
        "Expiration Date": "Not Found",
        "Passport Number": "Not Found",
        "Nationality": "Not Found"
    }

    # Clean the text and split into lines
    lines = [line.strip().replace(' ', '') for line in text.splitlines() if len(line.strip()) >= 30]
    
    # Debug logging
    logger.info(f"MRZ lines detected: {len(lines)}")
    for i, line in enumerate(lines):
        logger.info(f"Line {i}: {line}")
    
    if len(lines) >= 2:
        # Get the last two lines which should be the MRZ
        line1 = lines[-2]
        line2 = lines[-1]
        
        logger.info(f"Processing MRZ line 1: {line1}")
        logger.info(f"Processing MRZ line 2: {line2}")

        try:
            # Process first line (document type, country, name)
            if len(line1) >= 5 and (line1[0] == "P" or line1.startswith("P<")):
                data["Document Type"] = "Passport"  # More descriptive than just "P"
                
                # Extract country code (usually positions 2-5)
                country_match = re.search(r'P[<]?([A-Z]{3})', line1)
                if country_match:
                    data["Issuing Country"] = country_match.group(1)
                
                # Extract name parts - handle different formats
                name_part = line1[5:] if len(line1) > 5 else ""
                names = name_part.split("<<")
                surname = names[0].replace("<", " ").strip() if names else ""
                given = names[1].replace("<", " ").strip() if len(names) > 1 else ""
                data["Full Name"] = f"{given} {surname}".strip()
                
                logger.info(f"Extracted name: {data['Full Name']}")

            # Process second line (passport number, nationality, DOB, sex, expiry)
            if len(line2) >= 20:  # Ensure line is long enough for basic info
                # Try to extract passport number (usually first 9 chars)
                passport_match = re.search(r'^([A-Z0-9]{7,9})', line2)
                if passport_match:
                    data["Passport Number"] = passport_match.group(1)
                
                # Try to find nationality (usually after passport number)
                nationality_match = re.search(r'[0-9][A-Z]([A-Z]{3})', line2)
                if nationality_match:
                    data["Nationality"] = nationality_match.group(1)
                
                # Try to extract date of birth (6 digits after nationality)
                dob_match = re.search(r'[A-Z]{3}(\d{6})', line2)
                if dob_match:
                    dob = dob_match.group(1)
                    # Format date (YY-MM-DD to YYYY-MM-DD)
                    year_prefix = "19" if int(dob[0:2]) > 50 else "20"  # Assume 1950+ or 2000+
                    data["Date of Birth"] = f"{year_prefix}{dob[0:2]}-{dob[2:4]}-{dob[4:6]}"
                
                # Try to extract sex (usually 1 char after DOB)
                sex_match = re.search(r'\d{6}([MF])', line2)
                if sex_match:
                    data["Sex"] = sex_match.group(1)
                
                # Try to extract expiry date (6 digits after sex)
                expiry_match = re.search(r'[MF](\d{6})', line2)
                if expiry_match:
                    expiry = expiry_match.group(1)
                    # Format date (YY-MM-DD to YYYY-MM-DD)
                    year_prefix = "20"  # Assume expiry dates are in the 2000s
                    data["Expiration Date"] = f"{year_prefix}{expiry[0:2]}-{expiry[2:4]}-{expiry[4:6]}"
                
                logger.info(f"Extracted passport number: {data.get('Passport Number')}")
                logger.info(f"Extracted DOB: {data.get('Date of Birth')}")

            # Fallback: Try to extract data from the full text if MRZ parsing failed
            if all(value == "Not Found" for key, value in data.items() if key != "_validation"):
                logger.info("MRZ parsing failed, trying to extract from full text")
                
                # Set document type to Passport if found in text
                if "PASSPORT" in text.upper() or "PASSEPORT" in text.upper() or "PASAPORTE" in text.upper():
                    data["Document Type"] = "Passport"
                
                # Try to find passport number in the full text
                passport_match = re.search(r'(?:Passport|Number|No)[:\.\s]+([A-Z0-9]{7,9})', text, re.IGNORECASE)
                if passport_match:
                    data["Passport Number"] = passport_match.group(1)
                
                # Try to find name in the full text
                name_match = re.search(r'(?:Name|Nom|Surname)[:\.\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
                if name_match:
                    surname = name_match.group(1).strip()
                    
                    # Try to find given name
                    given_name_match = re.search(r'(?:Given|Prenom)[:\.\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
                    if given_name_match:
                        given_name = given_name_match.group(1).strip()
                        data["Full Name"] = f"{given_name} {surname}".strip()
                    else:
                        data["Full Name"] = surname
                
                # Try to find nationality in the full text
                nationality_match = re.search(r'(?:Nationality|Nationalité)[:\.\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
                if nationality_match:
                    data["Nationality"] = nationality_match.group(1).strip()
                
                # Try to find date of birth in the full text
                dob_match = re.search(r'(?:Date of birth|Date de naissance|Birth|DOB)[:\.\s]+(\d{1,2}[\/\.\s-]+\d{1,2}[\/\.\s-]+\d{4})', text, re.IGNORECASE)
                if dob_match:
                    dob = dob_match.group(1)
                    # Try to standardize the date format
                    dob_parts = re.split(r'[\/\.\s-]+', dob)
                    if len(dob_parts) == 3:
                        data["Date of Birth"] = f"{dob_parts[2]}-{dob_parts[1]}-{dob_parts[0]}"
                
                # Try to find sex in the full text
                sex_match = re.search(r'(?:Sex|Sexe)[:\.\s]+([MF])', text, re.IGNORECASE)
                if sex_match:
                    data["Sex"] = sex_match.group(1).upper()
                
                # Try to find expiry date in the full text
                expiry_match = re.search(r'(?:Date of expiry|Date d\'expiration|Expiry|Expiration)[:\.\s]+(\d{1,2}[\/\.\s-]+\d{1,2}[\/\.\s-]+\d{4})', text, re.IGNORECASE)
                if expiry_match:
                    expiry = expiry_match.group(1)
                    # Try to standardize the date format
                    expiry_parts = re.split(r'[\/\.\s-]+', expiry)
                    if len(expiry_parts) == 3:
                        data["Expiration Date"] = f"{expiry_parts[2]}-{expiry_parts[1]}-{expiry_parts[0]}"
                
                # Try to find issuing country in the full text
                country_match = re.search(r'(?:REPUBLIC|REPUBLIQUE|UNITED STATES)[E]?\s+(?:OF|DE)?\s+([A-Za-z\s]+)', text, re.IGNORECASE)
                if country_match:
                    country = country_match.group(1).strip()
                    if "UNITED STATES" in text.upper() or "USA" in text.upper() or "AMERICA" in text.upper():
                        data["Issuing Country"] = "USA"
                    else:
                        data["Issuing Country"] = country
                elif "FRANCE" in text.upper() or "FRANCAISE" in text.upper():
                    data["Issuing Country"] = "FRA"
                
                # Special case for USA passports
                if "UNITED STATES" in text.upper() or "USA" in text.upper() or "AMERICA" in text.upper():
                    data["Issuing Country"] = "USA"
                    data["Nationality"] = "USA"

        except Exception as e:
            logger.error(f"MRZ parsing error: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Validate the extracted data
    validation_results = {}
    
    # Only validate fields that were actually found
    if data["Date of Birth"] != "Not Found":
        dob_valid, dob_msg = validate_date(data["Date of Birth"])
        validation_results["Date of Birth"] = {"valid": dob_valid, "message": dob_msg}
    
    if data["Expiration Date"] != "Not Found":
        exp_valid, exp_msg = validate_date(data["Expiration Date"])
        validation_results["Expiration Date"] = {"valid": exp_valid, "message": exp_msg}
    
    if data["Issuing Country"] != "Not Found":
        country_valid, country_msg = validate_country_code(data["Issuing Country"])
        validation_results["Issuing Country"] = {"valid": country_valid, "message": country_msg}
    
    if data["Nationality"] != "Not Found":
        nationality_valid, nationality_msg = validate_country_code(data["Nationality"])
        validation_results["Nationality"] = {"valid": nationality_valid, "message": nationality_msg}
    
    if data["Passport Number"] != "Not Found":
        passport_valid, passport_msg = validate_passport_number(data["Passport Number"])
        validation_results["Passport Number"] = {"valid": passport_valid, "message": passport_msg}
    
    if data["Sex"] != "Not Found":
        sex_valid, sex_msg = validate_sex(data["Sex"])
        validation_results["Sex"] = {"valid": sex_valid, "message": sex_msg}
    
    # Add validation results to data
    data["_validation"] = validation_results
    
    return data

def detect_mrz_region(image_cv, use_client_side=False):
    """
    Detect MRZ region in passport image
    
    Args:
        image_cv: OpenCV image
        use_client_side: Whether to use client-side processing
        
    Returns:
        mrz_image: Image of MRZ region
        mrz_text: Extracted MRZ text
    """
    try:
        h, w = image_cv.shape[:2]
        
        # Try multiple approaches to find MRZ region
        
        # 1. Focus on bottom third of passport
        bottom_third = image_cv[int(h * 0.7):h, :]
        
        # Convert to grayscale
        if len(bottom_third.shape) == 3:
            gray = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
        else:
            gray = bottom_third
            
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        candidate_boxes = []
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            aspect_ratio = w_box / float(h_box)
            
            # MRZ lines typically have a width-to-height ratio > 4
            if w_box > 100 and 10 < h_box < 50 and aspect_ratio > 4:
                candidate_boxes.append((x, y, w_box, h_box))
        
        # Sort by y-coordinate (bottom to top)
        candidate_boxes = sorted(candidate_boxes, key=lambda x: x[1], reverse=True)
        
        # Try each candidate box
        for (x, y, w_box, h_box) in candidate_boxes[:5]:  # Check top 5 candidates
            roi = gray[y:y+h_box, x:x+w_box]
            
            # Enhance MRZ region
            roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            roi = cv2.GaussianBlur(roi, (3, 3), 0)
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use Tesseract with specific config for MRZ
            try:
                text_candidate = pytesseract.image_to_string(
                    roi, 
                    lang='eng', 
                    config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
                )
                
                # Check if this looks like MRZ text
                lines = [line for line in text_candidate.splitlines() if len(line.strip()) > 30]
                if len(lines) >= 2 and ('P<' in lines[0] or lines[0].startswith('P')):
                    logger.info(f"MRZ detected with {len(lines)} lines")
                    return roi, text_candidate
            except Exception as e:
                logger.error(f"Tesseract error: {e}")
                continue  # Try next candidate
        
        # If no good candidates found, try the full image as a fallback
        try:
            # Try OCR on the full image
            if len(image_cv.shape) == 3:
                full_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            else:
                full_gray = image_cv
                
            full_text = pytesseract.image_to_string(
                full_gray, 
                lang='eng',
                config='--psm 6'
            )
            
            # Look for MRZ-like patterns in the full text
            lines = full_text.splitlines()
            mrz_lines = []
            
            for i, line in enumerate(lines):
                if len(line.strip()) > 30 and ('P<' in line or line.strip().startswith('P')):
                    mrz_lines.append(line)
                    # Try to get the next line too if it exists
                    if i+1 < len(lines) and len(lines[i+1].strip()) > 30:
                        mrz_lines.append(lines[i+1])
                        break
            
            if len(mrz_lines) >= 2:
                logger.info("Found MRZ-like pattern in full image")
                return full_gray, "\n".join(mrz_lines)
            
            # If still no MRZ found, return the full text
            logger.info("No MRZ pattern found, returning full text")
            return full_gray, full_text
            
        except Exception as e:
            logger.error(f"Full image processing error: {e}")
            # Return the original grayscale image and an error message
            return gray, "⚠️ MRZ region not confidently detected"
            
    except Exception as e:
        logger.error(f"MRZ detection error: {e}")
        # Get detailed error explanation
        error_details = get_error_explanation("mrz_detection_failed", str(e))
        # Return a blank image and error message with explanation
        blank_image = np.zeros((100, 300), dtype=np.uint8)
        error_msg = f"⚠️ Error detecting MRZ: {str(e)}\n\nPossible causes: {', '.join(error_details['possible_causes'])}\n\nTry: {error_details['suggested_fixes'][0]}"
        return blank_image, error_msg

def detect_mrz_region_simple(image_cv):
    """
    A simplified version of MRZ detection that's less likely to crash
    
    Args:
        image_cv: OpenCV image
        
    Returns:
        mrz_text: Extracted MRZ text
    """
    try:
        # Convert to grayscale if needed
        if len(image_cv.shape) == 3:
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_cv
        
        # Try to use pytesseract directly on the image
        try:
            import pytesseract
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            logger.error(f"Pytesseract error: {e}")
            return "Error processing MRZ"
    except Exception as e:
        logger.error(f"Simple MRZ detection error: {e}")
        return "Error processing MRZ"
