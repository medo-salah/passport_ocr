
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
import io
import numpy as np
import cv2
import base64
import requests
from PIL import Image
import logging
import easyocr
import json
import time
import uuid
from datetime import datetime
from functools import lru_cache
import os
import sys
import concurrent.futures
from multiprocessing import cpu_count
import asyncio
from functools import partial
from pydantic_settings import BaseSettings

# Import processing modules from your Streamlit app
from config import LANGUAGE_MAP
from image_processing import preprocess_image
from mrz_processing import extract_mrz_fields, detect_mrz_region
from ner_processing import load_spacy_model, extract_with_ner
from utils.blur_detection import is_blurry
from utils.barcode_reader import extract_barcodes
from hologram_detection import detect_fake_document, verify_security_features
from document_analysis import detect_face, classify_document, detect_qr_codes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add model caching functions
@lru_cache(maxsize=1)
def get_ocr_reader(language="en", gpu=True):
    """Cache the EasyOCR reader to avoid reloading it for each request"""
    logger.info(f"Loading EasyOCR reader for language: {language}, GPU: {gpu}")
    return easyocr.Reader([language], gpu=gpu)

@lru_cache(maxsize=1)
def get_spacy_model():
    """Cache the spaCy NLP model to avoid reloading it for each request"""
    logger.info("Loading spaCy model")
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.error("Model 'en_core_web_sm' not found. Attempting to download...")
        try:
            # Try to download the model
            import subprocess
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            logger.info("Model downloaded successfully")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.error("Please run `python -m spacy download en_core_web_sm` manually")
            # Return a minimal model as fallback
            return spacy.blank("en")

# Add a function to preload models at startup
def preload_models():
    """Preload models at startup to improve response time for first request"""
    logger.info("Preloading models...")
    # Preload OCR reader
    get_ocr_reader()
    # Preload spaCy model
    get_spacy_model()
    logger.info("Models preloaded successfully")

# API configuration
class Settings(BaseSettings):
    max_workers: int = 8
    use_gpu: bool = True
    port: int = 8000
    host: str = "0.0.0.0"
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"

# Create settings instance
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Passport OCR API",
    description="""
    API for passport OCR and document verification.
    
    This API provides endpoints for:
    - Processing passport and ID document images
    - Extracting text and structured data from documents
    - Detecting faces in documents
    - Verifying document authenticity
    - Classifying document types
    
    For more information, visit our documentation at https://example.com/docs
    """,
    version="1.0.0",
    contact={
        "name": "Support Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "Proprietary",
        "url": "https://example.com/license",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class PreprocessingProfile(BaseModel):
    name: str
    description: str
    denoise: bool = True
    contrast: bool = True
    sharpen: bool = True
    threshold: bool = False
    deskew: bool = False
    additional_params: Dict[str, Any] = Field(default_factory=dict)

# Define common preprocessing profiles
PREPROCESSING_PROFILES = {
    "default": PreprocessingProfile(
        name="default",
        description="Balanced profile for most documents",
        denoise=True,
        contrast=True,
        sharpen=True,
        threshold=False,
        deskew=True
    ),
    "passport_mrz": PreprocessingProfile(
        name="passport_mrz",
        description="Optimized for passport MRZ regions",
        denoise=True,
        contrast=True,
        sharpen=True,
        threshold=True,
        deskew=True,
        additional_params={"threshold_method": "adaptive"}
    ),
    "id_card": PreprocessingProfile(
        name="id_card",
        description="Optimized for ID cards",
        denoise=True,
        contrast=True,
        sharpen=False,
        threshold=False,
        deskew=True
    ),
    "low_quality": PreprocessingProfile(
        name="low_quality",
        description="For low quality or blurry images",
        denoise=True,
        contrast=True,
        sharpen=True,
        threshold=False,
        deskew=True,
        additional_params={"super_resolution": True}
    ),
    "dark_document": PreprocessingProfile(
        name="dark_document",
        description="For dark or underexposed documents",
        denoise=True,
        contrast=True,
        sharpen=True,
        threshold=False,
        deskew=True,
        additional_params={"gamma_correction": 1.5}
    )
}

class ProcessingOptions(BaseModel):
    language: str = Field("eng", description="OCR language code")
    enable_face_detection: bool = Field(False, description="Enable face detection")
    enable_document_classification: bool = Field(True, description="Auto-detect document type")
    enable_authenticity_check: bool = Field(False, description="Check document authenticity")
    enable_preprocessing: bool = Field(True, description="Apply image preprocessing")
    preprocessing_profile: Optional[str] = Field(
        None, 
        description="Preprocessing profile name (default, passport_mrz, id_card, low_quality, dark_document)"
    )
    preprocessing_options: Dict[str, Any] = Field(
        default_factory=lambda: {
            "denoise": True,
            "contrast": True,
            "sharpen": True,
            "threshold": False,
        },
        description="Image preprocessing options",
    )

class ProcessingResult(BaseModel):
    request_id: str
    timestamp: str
    document_data: Dict[str, Any]
    validation: Optional[Dict[str, Any]] = None
    authenticity: Optional[Dict[str, Any]] = None
    face_detection: Optional[Dict[str, Any]] = None
    document_type: Optional[Dict[str, Any]] = None
    processing_time: float

class WebhookConfig(BaseModel):
    url: str
    event: str = "document_processed"
    secret: Optional[str] = None

class BatchProcessingRequest(BaseModel):
    document_urls: List[str] = Field(..., description="List of document image URLs to process")
    options: Optional[ProcessingOptions] = Field(None, description="Processing options")
    callback_url: Optional[str] = Field(None, description="Webhook URL for results")

class BatchProcessingResponse(BaseModel):
    batch_id: str
    status: str
    message: str
    estimated_time: float

class ComparisonRequest(BaseModel):
    extracted_data: Dict[str, Any] = Field(..., description="Data extracted from document")
    reference_data: Dict[str, Any] = Field(..., description="Reference data to compare against")

class ComparisonResult(BaseModel):
    matches: Dict[str, bool]
    match_score: float
    overall_match: bool
    mismatched_fields: List[str]

# Define API endpoints
@app.post("/process", response_model=ProcessingResult, 
          responses={
              200: {"description": "Document processed successfully"},
              400: {"description": "Invalid input parameters"},
              500: {"description": "Processing error"}
          },
          tags=["Document Processing"])
async def process_document(
    file: UploadFile = File(..., description="The document image file (JPG, PNG)"),
    options: Optional[str] = Form(None, description="JSON string with processing options"),
    webhook: Optional[str] = Form(None, description="Optional webhook URL to receive results asynchronously"),
):
    """
    Process a passport or ID document image
    
    ## Description
    This endpoint processes an uploaded document image and extracts information from it.
    The processing can include OCR, MRZ extraction, face detection, document classification,
    and authenticity verification depending on the options provided.
    
    ## Parameters
    - **file**: The document image file (JPG, PNG)
    - **options**: JSON string with processing options (see example below)
    - **webhook**: Optional webhook URL to receive results asynchronously
    
    ## Example Options
    ```json
    {
        "language": "eng",
        "enable_face_detection": true,
        "enable_document_classification": true,
        "enable_authenticity_check": false,
        "enable_preprocessing": true,
        "preprocessing_options": {
            "denoise": true,
            "contrast": true,
            "sharpen": true,
            "threshold": false
        }
    }
    ```
    
    ## Returns
    A JSON object containing the extracted document data, validation results,
    and other information based on the enabled processing options.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Parse options
    if options:
        try:
            options_dict = json.loads(options)
            processing_options = ProcessingOptions(**options_dict)
        except Exception as e:
            logger.error(f"Error parsing options: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid options format: {str(e)}")
    else:
        processing_options = ProcessingOptions()
    
    # Parse webhook
    webhook_config = None
    if webhook:
        try:
            webhook_dict = json.loads(webhook)
            webhook_config = WebhookConfig(**webhook_dict)
        except Exception as e:
            logger.error(f"Error parsing webhook config: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid webhook format: {str(e)}")
    
    # Process the image
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Check if image is blurry
        blurry, blur_score = is_blurry(image)
        if blurry:
            logger.warning(f"Blurry image detected (score: {blur_score})")
        
        # Preprocess image if enabled
        if processing_options.enable_preprocessing:
            # Use profile-based preprocessing if profile is specified
            if processing_options.preprocessing_profile:
                processed_image = apply_preprocessing_profile(
                    image,
                    profile_name=processing_options.preprocessing_profile,
                    custom_options=processing_options.preprocessing_options
                )
            else:
                # Use legacy preprocessing with explicit options
                processed_image = preprocess_image(
                    image,
                    denoise=processing_options.preprocessing_options.get("denoise", True),
                    contrast=processing_options.preprocessing_options.get("contrast", True),
                    sharpen=processing_options.preprocessing_options.get("sharpen", True),
                    threshold=processing_options.preprocessing_options.get("threshold", False),
                )
        else:
            processed_image = image
        
        # Convert to OpenCV format for processing
        processed_cv = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        
        # Run tasks in parallel
        async def run_parallel_tasks():
            # Create tasks for parallel execution
            tasks = []
            
            # Document classification task
            if processing_options.enable_document_classification:
                tasks.append(asyncio.to_thread(classify_document, processed_image))
            else:
                tasks.append(None)
            
            # Face detection task
            if processing_options.enable_face_detection:
                tasks.append(asyncio.to_thread(detect_face, processed_image))
            else:
                tasks.append(None)
            
            # OCR task
            ocr_task = asyncio.to_thread(
                get_ocr_reader(
                    language=LANGUAGE_MAP.get(processing_options.language, "en"),
                    gpu=os.environ.get("USE_GPU", "1") == "1"
                ).readtext,
                np.array(processed_image),
                detail=1
            )
            tasks.append(ocr_task)
            
            # MRZ detection task
            tasks.append(asyncio.to_thread(detect_mrz_region, processed_cv))
            
            # Barcode detection task
            tasks.append(asyncio.to_thread(extract_barcodes, processed_cv))
            
            # QR code detection task
            tasks.append(asyncio.to_thread(detect_qr_codes, processed_image))
            
            # Authenticity check task
            if processing_options.enable_authenticity_check:
                tasks.append(asyncio.to_thread(detect_fake_document, processed_image))
                
                # Apply security-focused preprocessing for better feature detection
                security_image = apply_preprocessing_profile(
                    processed_image,
                    profile_name="passport_mrz" if tasks[0] and (await tasks[0])[0] == "Passport" else "default"
                )
                tasks.append(asyncio.to_thread(verify_security_features, security_image))
            else:
                tasks.append(None)
                tasks.append(None)
            
            # Execute all tasks in parallel and return results
            return await asyncio.gather(*[task for task in tasks if task is not None])
        
        # Run parallel tasks
        parallel_results = await run_parallel_tasks()
        
        # Extract results
        result_index = 0
        
        # Document classification result
        document_type_info = None
        if processing_options.enable_document_classification:
            doc_type, confidence = parallel_results[result_index]
            document_type_info = {
                "type": doc_type,
                "confidence": confidence
            }
            result_index += 1
        
        # Face detection result
        face_info = None
        if processing_options.enable_face_detection:
            face_img, face_coords = parallel_results[result_index]
            if face_img is not None:
                # Convert face image to base64 for response
                face_pil = Image.fromarray(face_img)
                buffered = io.BytesIO()
                face_pil.save(buffered, format="JPEG")
                face_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                face_info = {
                    "detected": True,
                    "coordinates": face_coords,
                    "image_base64": face_base64
                }
            else:
                face_info = {
                    "detected": False
                }
            result_index += 1
        
        # OCR result
        ocr_results = parallel_results[result_index]
        text = "\n".join([r[1] for r in ocr_results])
        result_index += 1
        
        # MRZ result
        mrz_image, mrz_text = parallel_results[result_index]
        result_index += 1
        
        # Barcode result
        barcodes = parallel_results[result_index]
        result_index += 1
        
        # QR code result
        qr_data = parallel_results[result_index]
        result_index += 1
        
        # Data extraction
        nlp = get_spacy_model()
        data = extract_mrz_fields(mrz_text)
        ner_data = extract_with_ner(text, nlp)
        
        # Only update fields that are "Not Found" with NER data
        for key, value in ner_data.items():
            if key not in data or data[key] == "Not Found":
                data[key] = value
        
        # Add barcode data
        if barcodes:
            data["Barcodes"] = barcodes
        
        # Add QR code data
        if qr_data:
            data["QR_Codes"] = qr_data
        
        # Authenticity check result
        authenticity_info = None
        if processing_options.enable_authenticity_check:
            _, detection_data = parallel_results[result_index]
            result_index += 1
            security_result, authenticity_score = parallel_results[result_index]
            result_index += 1
            
            authenticity_info = {
                "score": authenticity_score,
                "assessment": security_result["assessment"],
                "issues": detection_data["issues"],
                "fake_detected": detection_data["fake_detected"],
                "confidence": detection_data["confidence"]
            }
        
        # Prepare response
        processing_time = time.time() - start_time
        result = ProcessingResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            document_data=data,
            validation=data.get("_validation"),
            authenticity=authenticity_info,
            face_detection=face_info,
            document_type=document_type_info,
            processing_time=processing_time
        )
        
        # Send webhook if configured
        if webhook_config:
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                send_webhook,
                webhook_config.url,
                result.dict(),
                webhook_config.secret
            )
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

async def send_webhook(url: str, data: dict, secret: Optional[str] = None):
    """Send webhook with processing results"""
    try:
        headers = {"Content-Type": "application/json"}
        if secret:
            import hmac
            import hashlib
            
            # Create signature
            message = json.dumps(data).encode()
            signature = hmac.new(
                secret.encode(), message, hashlib.sha256
            ).hexdigest()
            headers["X-Signature"] = signature
        
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"Webhook sent to {url}, status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending webhook: {e}")

# Define webhook for receiving document processing requests
class DocumentRequest(BaseModel):
    document_url: str
    options: Optional[ProcessingOptions] = None
    callback_url: Optional[str] = None

@app.webhooks.post("document-processing-request")
def document_processing_request(body: DocumentRequest):
    """
    When you want to process a document, send a POST request with this data
    to the URL that you register for the event 'document-processing-request'.
    
    We'll download the document from the provided URL, process it according
    to the options, and send the results to your callback URL if provided.
    """
    pass

@app.post("/batch-process", response_model=BatchProcessingResponse, tags=["Document Processing"])
async def batch_process_documents(request: BatchProcessingRequest):
    """
    Process multiple documents in batch mode
    
    This endpoint allows you to submit multiple document URLs for processing.
    The processing will be done asynchronously, and results will be sent to the
    provided callback URL when completed.
    
    ## Parameters
    - **document_urls**: List of URLs pointing to document images
    - **options**: Processing options (same as for single document processing)
    - **callback_url**: Webhook URL to receive results when processing is complete
    
    ## Returns
    A batch ID and status information that can be used to check the progress
    of the batch processing job.
    """
    batch_id = str(uuid.uuid4())
    num_documents = len(request.document_urls)
    estimated_time = num_documents * 2.5  # Rough estimate: 2.5 seconds per document
    
    # Start background processing
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        process_document_batch,
        batch_id,
        request.document_urls,
        request.options,
        request.callback_url
    )
    
    return BatchProcessingResponse(
        batch_id=batch_id,
        status="processing",
        message=f"Processing {num_documents} documents in batch mode",
        estimated_time=estimated_time
    )

async def process_document_batch(
    batch_id: str,
    document_urls: List[str],
    options: Optional[ProcessingOptions],
    callback_url: Optional[str]
):
    """Process a batch of documents in the background using parallel processing"""
    from parallel_processing import parallel_process
    
    # Define a function to process a single document URL
    def process_single_document(url):
        try:
            # Download image from URL
            response = requests.get(url)
            if response.status_code != 200:
                return {
                    "url": url,
                    "status": "error",
                    "error": f"Failed to download image: HTTP {response.status_code}"
                }
            
            # Process image
            image = Image.open(io.BytesIO(response.content))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Check if image is blurry
            blurry, blur_score = is_blurry(image)
            if blurry:
                return {
                    "url": url,
                    "status": "error",
                    "error": f"Image too blurry (score: {blur_score})"
                }
            
            # Preprocess image
            if options and options.enable_preprocessing:
                if options.preprocessing_profile:
                    processed_image = apply_preprocessing_profile(
                        image,
                        profile_name=options.preprocessing_profile,
                        custom_options=options.preprocessing_options
                    )
                else:
                    processed_image = preprocess_image(
                        image,
                        denoise=options.preprocessing_options.get("denoise", True),
                        contrast=options.preprocessing_options.get("contrast", True),
                        sharpen=options.preprocessing_options.get("sharpen", True),
                        threshold=options.preprocessing_options.get("threshold", False),
                    )
            else:
                processed_image = image
            
            # Convert to OpenCV format
            processed_cv = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            
            # OCR processing
            language = options.language if options else "eng"
            reader = get_ocr_reader(
                language=LANGUAGE_MAP.get(language, "en"),
                gpu=os.environ.get("USE_GPU", "1") == "1"
            )
            results = reader.readtext(np.array(processed_image), detail=1)
            text = "\n".join([r[1] for r in results])
            
            # MRZ processing
            mrz_image, mrz_text = detect_mrz_region(processed_cv)
            
            # Data extraction
            nlp = get_spacy_model()
            data = extract_mrz_fields(mrz_text)
            ner_data = extract_with_ner(text, nlp)
            
            # Only update fields that are "Not Found" with NER data
            for key, value in ner_data.items():
                if key not in data or data[key] == "Not Found":
                    data[key] = value
            
            # Extract barcodes if any
            barcodes = extract_barcodes(processed_cv)
            if barcodes:
                data["Barcodes"] = barcodes
            
            # Extract QR codes if any
            qr_data = detect_qr_codes(processed_image)
            if qr_data:
                data["QR_Codes"] = qr_data
            
            # Document classification if enabled
            if options and options.enable_document_classification:
                doc_type, confidence = classify_document(processed_image)
                data["Document_Type"] = {
                    "type": doc_type,
                    "confidence": confidence
                }
            
            # Face detection if enabled
            if options and options.enable_face_detection:
                face_img, face_coords = detect_face(processed_image)
                if face_img is not None:
                    # Convert face image to base64 for response
                    face_pil = Image.fromarray(face_img)
                    buffered = io.BytesIO()
                    face_pil.save(buffered, format="JPEG")
                    face_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    data["Face"] = {
                        "detected": True,
                        "coordinates": face_coords,
                        "image_base64": face_base64
                    }
                else:
                    data["Face"] = {
                        "detected": False
                    }
            
            return {
                "url": url,
                "status": "success",
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error processing document {url}: {e}")
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }
    
    # Process all documents in parallel
    results = await async_parallel_process(
        document_urls,
        process_single_document,
        max_workers=min(len(document_urls), 8)  # Limit to 8 workers max
    )
    
    # Send results to callback URL if provided
    if callback_url:
        try:
            requests.post(
                callback_url,
                json={
                    "batch_id": batch_id,
                    "status": "completed",
                    "results": results
                }
            )
        except Exception as e:
            logger.error(f"Error sending batch results to callback URL: {e}")

@app.post("/compare-data", response_model=ComparisonResult, tags=["Document Verification"])
async def compare_document_data(request: ComparisonRequest):
    """
    Compare extracted document data with reference data
    
    This endpoint compares data extracted from a document with reference data
    to verify if the information matches. It returns a detailed comparison result
    with field-by-field matching and an overall match score.
    
    ## Parameters
    - **extracted_data**: Data extracted from the document
    - **reference_data**: Reference data to compare against (e.g., from a database)
    
    ## Returns
    A comparison result with field-by-field matching and an overall match score.
    """
    matches = {}
    mismatched_fields = []
    
    # Compare each field in reference data with extracted data
    for field, ref_value in request.reference_data.items():
        if field in request.extracted_data:
            # Case-insensitive comparison for text fields
            if isinstance(ref_value, str) and isinstance(request.extracted_data[field], str):
                matches[field] = ref_value.lower() == request.extracted_data[field].lower()
            else:
                matches[field] = ref_value == request.extracted_data[field]
            
            if not matches[field]:
                mismatched_fields.append(field)
        else:
            matches[field] = False
            mismatched_fields.append(field)
    
    # Calculate match score (percentage of matching fields)
    match_score = sum(1 for match in matches.values() if match) / len(matches) if matches else 0
    
    # Determine overall match (e.g., if match score is above 80%)
    overall_match = match_score >= 0.8
    
    return ComparisonResult(
        matches=matches,
        match_score=match_score,
        overall_match=overall_match,
        mismatched_fields=mismatched_fields
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "ok", "version": "1.0.0"}

# Get available preprocessing profiles
@app.get("/preprocessing-profiles", tags=["Configuration"])
async def get_preprocessing_profiles():
    """
    Get available preprocessing profiles
    
    Returns a list of available preprocessing profiles that can be used
    with the document processing endpoints.
    """
    return {
        "profiles": [
            {
                "name": profile.name,
                "description": profile.description,
                "options": {
                    "denoise": profile.denoise,
                    "contrast": profile.contrast,
                    "sharpen": profile.sharpen,
                    "threshold": profile.threshold,
                    "deskew": profile.deskew,
                    **profile.additional_params
                }
            }
            for profile in PREPROCESSING_PROFILES.values()
        ]
    }

def check_dependencies():
    """Check if all required dependencies are installed and available"""
    missing_deps = []
    
    # Check for spaCy model
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            missing_deps.append("spaCy model 'en_core_web_sm'")
    except ImportError:
        missing_deps.append("spacy")
    
    # Check for EasyOCR
    try:
        import easyocr
    except ImportError:
        missing_deps.append("easyocr")
    
    # Check for OpenCV
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    # Check for TensorFlow (optional)
    try:
        import tensorflow
        tf_available = True
    except ImportError:
        tf_available = False
        logger.warning("TensorFlow not available. Some features may be limited.")
    
    # Report missing dependencies
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install missing dependencies before running the API")
        return False
    
    return True

# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    # Check dependencies
    if not check_dependencies():
        logger.warning("Starting API with limited functionality due to missing dependencies")
    
    # Preload models to improve startup time
    try:
        preload_models()
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
        logger.warning("API will start but first requests may be slower")
    
    # Configure environment variables
    os.environ["USE_GPU"] = "1" if settings.use_gpu else "0"
    
    # Log configuration
    logger.info(f"Starting API with configuration:")
    logger.info(f"  Host: {settings.host}")
    logger.info(f"  Port: {settings.port}")
    logger.info(f"  Max workers: {settings.max_workers}")
    logger.info(f"  GPU enabled: {settings.use_gpu}")
    
    # Start the server
    uvicorn.run(app, host=settings.host, port=settings.port)



'''

To use this API integration:

Run your Streamlit app as usual
Separately run the API server with python api.py
The API will be available at http://localhost:8000
Access the API documentation at http://localhost:8000/docs

'''
