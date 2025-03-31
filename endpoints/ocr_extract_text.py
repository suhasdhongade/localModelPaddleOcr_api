from fastapi import APIRouter, UploadFile, File
from paddleocr import PaddleOCR
import numpy as np
import cv2

router = APIRouter()

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    det_model_dir="./models/det",
    rec_model_dir="./models/rec",
    cls_model_dir="./models/cls"
)

@router.post("/ocr/extract_text_from_image", summary="Extract text from image and return full text and lines with confidence score")
async def get_text(file: UploadFile = File(...)):
    """
    This endpoint takes an image file as input, performs OCR to extract text,
    and returns the extracted text.
    
    - **file**: Upload an image file.
    - **Returns**: Extracted text, confidence scores. Highest confidence score is better .
    - **Returns**: Extracted text, confidence scores, and the path to the annotated image.
    - **1.0 (or close to 1) Confidence Score**: The model is very confident that the detected text is accurate.
    - **0.5 Confidence Score**: The model is very confident that the detected text is accurate.
    - **0.0 (or close to 0) Confidence Score**:  The model is uncertain about the detected text
    """
    contents = await file.read()

    # Convert image to numpy array
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform OCR
    results = ocr.ocr(image, cls=True)

    extracted_text = []
    line_items = []
    full_text = ""
    total_confidence = 0
    num_texts = 0

    for result in results:
        for line in result:
            text = line[1][0]
            confidence = line[1][1]
            extracted_text.append(text)
            full_text += text + " "
            total_confidence += confidence
            num_texts += 1
            line_items.append({"text": text, "confidence": confidence})
    
    avg_confidence = total_confidence / num_texts if num_texts > 0 else 0
    
    return {
        "full_text": full_text.strip(),
        "full_text_average_confidence": avg_confidence,
        "line_items": line_items
    }
