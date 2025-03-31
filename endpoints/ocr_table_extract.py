from fastapi import APIRouter, UploadFile, File, HTTPException
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

@router.post("/ocr/extract_table_from_image", summary="Extract tables from an image.Not tested fully")
async def extract_table(file: UploadFile = File(...)):
    """
    Extract tables from an image and return structured data.
    """
    contents = await file.read()
    try:
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Perform OCR
    results = ocr.ocr(image, cls=True)

    # Extract bounding boxes and text
    table_data = []
    for result in results:
        for line in result:
            box = line[0]
            text = line[1][0]
            table_data.append({"box": box, "text": text})

    return {"table": table_data}