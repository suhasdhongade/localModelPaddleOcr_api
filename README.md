# OCR API with PaddleOCR

This project provides a FastAPI-based OCR (Optical Character Recognition) API using PaddleOCR. The API extracts text from images, provides confidence scores, generates annotated images with bounding boxes and text lists, and extracts tables from images.

## Features

- Extract text from images with confidence scores.
- Generate annotated images with bounding boxes and text lists.
- Save annotated images to a specified path.
- Extract tables from images in a structured format.
- Supports classification, detection, and recognition models.

## Endpoints

### 1. `/ocr/ExtractTextFromImage`

- **Method**: `POST`
- **Description**: Extracts text from an image and returns the full text, line items, and confidence scores.
- **Input**: Image file.
- **Output**: JSON with extracted text, average confidence, and line items.

### 2. `/ocr/GetAnnotedImage`

- **Method**: `POST`
- **Description**: Extracts text from an image and returns an annotated image with bounding boxes and text numbers.
- **Input**: Image file.
- **Output**: Annotated image.

### 3. `/ocr/`

- **Method**: `POST`
- **Description**: Extracts text from an image, saves the annotated image to a specified path, and returns the extracted text and confidence scores.
- **Input**: Image file.
- **Output**: JSON with extracted text, average confidence, line items, and the path to the saved image.

### 4. `/ocr/ExtractTableFromImage`

- **Method**: `POST`
- **Description**: Extracts tables from an image and returns the extracted table data in a structured format.
- **Input**: Image file.
- **Output**: JSON with extracted table data as a list of rows and columns.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suhasdhongade/localModelPaddleOcr_api.git
   cd localModelPaddleOcr_api
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   uvicorn app:app --reload
   ```

## Project Structure

.
├── [app.py](http://_vscodecontentref_/1) # Main FastAPI application
├── models/ # Directory containing PaddleOCR model files
│ ├── cls/
│ ├── det/
│ └── rec/
├── uploads/ # Directory for uploaded files
├── [requirements.txt](http://_vscodecontentref_/2) # Python dependencies
├── [README.md](http://_vscodecontentref_/3) # Project documentation
└── .gitignore # Git ignore file

GitHub Copilot
Updated README.md
Create a virtual environment and activate it:

Install dependencies:

Download the PaddleOCR model files and place them in the models/ directory:

models/cls/
models/det/
models/rec/

Running the Application
Start the FastAPI server: uvicorn app:app --reload

Open your browser and navigate to:

API Documentation: http://127.0.0.1:8000/docs
Redoc Documentation: http://127.0.0.1:8000/redoc
Directory Structure

Dependencies
FastAPI
PaddleOCR
OpenCV
NumPy
Pillow

Notes
Ensure the FONT_PATH in app.py points to a valid .ttf font file on your system.
Modify the output_path in the /ocr/ endpoint to specify where annotated images should be saved.

Contact
Author: Suhas Dhongade
Email: email@example.com
