# Passport OCR API

API for passport OCR and document verification.

## Setup

1. Install Python 3.8 or higher
2. Clone this repository
3. Run the setup script:

```bash
python setup.py
```

Or install dependencies manually:

```bash
pip install -r requirements-api.txt
python -m spacy download en_core_web_sm
```

## Running the API

```bash
python api.py
```

The API will be available at http://localhost:8000

## API Documentation

Once the API is running, you can access the interactive documentation at:
http://localhost:8000/docs

## Environment Variables

- `USE_GPU`: Set to "1" to enable GPU acceleration (default) or "0" to disable
- `PORT`: Set the port for the API server (default: 8000)
