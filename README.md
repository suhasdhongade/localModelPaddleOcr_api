# OCR API with PaddleOCR

This project provides a FastAPI-based OCR (Optical Character Recognition) API using PaddleOCR. The API extracts text from images, provides confidence scores, and generates annotated images with bounding boxes and text lists.

## Features

- Extract text from images with confidence scores.
- Generate annotated images with bounding boxes and text lists.
- Save annotated images to a specified path.
- Supports classification, detection, and recognition models.

## Endpoints

### 1. `/ocr/text`

- **Method**: `POST`
- **Description**: Extracts text from an image and returns the full text, line items, and confidence scores.
- **Input**: Image file.
- **Output**: JSON with extracted text, average confidence, and line items.

### 2. `/ocr/annotedimage`

- **Method**: `POST`
- **Description**: Extracts text from an image and returns an annotated image with bounding boxes and text numbers.
- **Input**: Image file.
- **Output**: Annotated image.

### 3. `/ocr/`

- **Method**: `POST`
- **Description**: Extracts text from an image, saves the annotated image to a specified path, and returns the extracted text and confidence scores.
- **Input**: Image file.
- **Output**: JSON with extracted text, average confidence, line items, and the path to the saved image.

### 4. `/ocr/imageAutoGrow`

- **Method**: `POST`
- **Description**: Extracts text from an image and returns an annotated image with bounding boxes and text numbers.
- **Input**: Image file.
- **Output**: Annotated image.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suhasdhongade/localModelPaddleOcr_api.git
   cd your-repo-name
   ```
