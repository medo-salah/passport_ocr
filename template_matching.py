import cv2
import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PassportTemplate:
    def __init__(self, name, template_image, regions, country_code=None):
        """
        Initialize a passport template
        
        Args:
            name: Template name (e.g., "EU Passport")
            template_image: Path to template image or numpy array
            regions: Dictionary of regions to extract (name -> [x, y, w, h])
            country_code: Optional 3-letter country code
        """
        self.name = name
        self.country_code = country_code
        
        # Load template image if path is provided
        if isinstance(template_image, str) or isinstance(template_image, Path):
            self.template = cv2.imread(str(template_image))
            if self.template is None:
                raise ValueError(f"Could not load template image: {template_image}")
        else:
            self.template = template_image
            
        # Store extraction regions
        self.regions = regions
    
    def get_region(self, image, region_name):
        """Extract a specific region from an image based on template matching"""
        if region_name not in self.regions:
            logger.warning(f"Region '{region_name}' not defined in template '{self.name}'")
            return None
            
        x, y, w, h = self.regions[region_name]
        return image[y:y+h, x:x+w]

def load_templates():
    """Load all passport templates from the templates directory"""
    templates = []
    template_dir = Path("templates")
    
    if not template_dir.exists():
        logger.warning(f"Template directory not found: {template_dir}")
        return templates
    
    # Example template definitions
    # In a real app, these would be loaded from configuration files
    templates.append(PassportTemplate(
        "EU Passport",
        "templates/eu_passport.jpg",
        {
            "mrz_region": [50, 680, 900, 100],
            "photo_region": [50, 180, 300, 400],
            "name_region": [400, 200, 500, 50],
            "birth_date_region": [400, 300, 200, 50],
            "expiry_date_region": [400, 400, 200, 50],
            "passport_number_region": [400, 500, 200, 50],
            "signature_region": [400, 600, 500, 70]  # Add signature region
        },
        "EUR"
    ))
    
    templates.append(PassportTemplate(
        "US Passport",
        "templates/us_passport.jpg",
        {
            "mrz_region": [50, 700, 900, 100],
            "photo_region": [50, 150, 300, 400],
            "name_region": [400, 180, 500, 50],
            "birth_date_region": [400, 280, 200, 50],
            "expiry_date_region": [400, 380, 200, 50],
            "passport_number_region": [400, 480, 200, 50],
            "signature_region": [400, 580, 500, 70]  # Add signature region
        },
        "USA"
    ))
    
    return templates

def match_template(image, templates, threshold=0.5):
    """
    Match an image against available templates
    
    Args:
        image: Input image
        templates: List of PassportTemplate objects
        threshold: Matching threshold (0-1)
        
    Returns:
        best_template: Best matching template or None
        confidence: Matching confidence score
    """
    best_match = None
    best_score = 0
    
    # Convert image to grayscale for matching
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize image to a standard size for matching
    resized = cv2.resize(gray, (1000, 800))
    
    for template in templates:
        # Convert template to grayscale
        if len(template.template.shape) == 3:
            template_gray = cv2.cvtColor(template.template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.template
            
        # Resize template to match input image size
        template_resized = cv2.resize(template_gray, (1000, 800))
        
        # Perform template matching
        result = cv2.matchTemplate(resized, template_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        logger.info(f"Template '{template.name}' match score: {max_val:.4f}")
        
        if max_val > best_score:
            best_score = max_val
            best_match = template
    
    # Return best match if score exceeds threshold
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, best_score

def extract_regions_from_template(image, template):
    """
    Extract all defined regions from an image based on a template
    
    Args:
        image: Input image
        template: PassportTemplate object
        
    Returns:
        Dictionary of region name -> extracted image
    """
    regions = {}
    
    # Resize image to match template size
    h, w = template.template.shape[:2]
    resized = cv2.resize(image, (w, h))
    
    for region_name, coords in template.regions.items():
        x, y, w, h = coords
        regions[region_name] = resized[y:y+h, x:x+w]
    
    return regions
