# OCR API with PaddleOCR

This project provides a FastAPI-based OCR (Optical Character Recognition) API utilizing PaddleOCR. The API enables users to extract text from images, obtain confidence scores, generate annotated images with bounding boxes and text lists, and extract tables from images.

## Features

- **Text Extraction**: Retrieve text from images along with confidence scores.
- **Annotated Images**: Generate images annotated with bounding boxes and corresponding text.
- **Table Extraction**: Extract tables from images in a structured format.
- **Model Support**: Supports classification, detection, and recognition models.

## Endpoints

1. **`/ocr/ExtractTextFromImage`**

   - **Method**: `POST`
   - **Description**: Extracts text from an image and returns the full text, line items, and confidence scores.
   - **Input**: Image file.
   - **Output**: JSON containing extracted text, average confidence, and line items.

2. **`/ocr/GetAnnotedImage`**

   - **Method**: `POST`
   - **Description**: Extracts text from an image and returns an annotated image with bounding boxes and text lists.
   - **Input**: Image file.
   - **Output**: Annotated image file.

3. **`/ocr/ExtractTableFromImage`**
   - **Method**: `POST`
   - **Description**: Extracts tables from an image and returns structured data.
   - **Input**: Image file.
   - **Output**: JSON containing extracted table data.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/suhasdhongade/localModelPaddleOcr_api.git
   cd localModelPaddleOcr_api
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.6 or later installed. It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   Install the required packages:

   ```bash
   pip install -r Requirements.txt
   ```

3. **Download and Place PaddleOCR Models**:
   Download the necessary PaddleOCR model files and place them in the `models` directory. Ensure the directory structure is as follows:
   ```
   models/
   ├── det
   │   └── det_model
   ├── rec
   │   └── rec_model
   └── cls
       └── cls_model
   ```
   Replace `det_model`, `rec_model`, and `cls_model` with the appropriate model files.

## Usage

1. **Start the FastAPI Server**:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   The API will be accessible at `http://0.0.0.0:8000`.

2. **Access the API Documentation**:
   Navigate to `http://0.0.0.0:8000/docs` in your browser to view the interactive API documentation provided by Swagger UI.

3. **Make API Requests**:
   Use tools like [Postman](https://www.postman.com/) or `curl` to send requests to the API endpoints. For example, to extract text from an image:
   ```bash
   curl -X 'POST' \
     'http://0.0.0.0:8000/ocr/ExtractTextFromImage' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@path_to_your_image.jpg'
   ```

## Configuration

- **Model Paths**: Update the `app.py` file to specify the paths to your local PaddleOCR models:

  ```python
  ocr = PaddleOCR(
      det_model_dir='models/det/det_model',
      rec_model_dir='models/rec/rec_model',
      cls_model_dir='models/cls/cls_model',
      use_angle_cls=True,
      lang='en'
  )
  ```

  Ensure that the paths correspond to the locations where you've placed the model files.

- **Logging**: The application uses Python's built-in `logging` module. Configure the logging level and format as needed in the `app.py` file:
  ```python
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
  ```

## Dependencies

- **FastAPI**: Web framework for building APIs with Python.
- **Uvicorn**: ASGI server for serving FastAPI applications.
- **PaddleOCR**: OCR tool based on PaddlePaddle.
- **Pillow**: Image processing library.
- **Python-Multipart**: Required for handling multipart/form-data requests.

## Notes

- **Model Files**: The PaddleOCR models are not included in this repository due to their size. Ensure you download the appropriate models and place them in the `models` directory as specified.
- **Error Handling**: The API includes basic error handling. For production use, consider implementing more robust error handling and validation.
- **Performance**: Depending on the size and complexity of the input images, the OCR process may take some time. Optimize the models and server resources as needed.

## References

- **PaddleOCR Documentation**: [https://paddlepaddle.github.io/PaddleOCR/](https://paddlepaddle.github.io/PaddleOCR/)
- **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

---

_This README provides an overview of the OCR API project, including its features, installation steps, usage instructions, and configuration details. Ensure you have the necessary PaddleOCR models and dependencies installed to use the API effectively._
