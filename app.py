from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import shutil
import os
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI()

# Initialize PaddleOCR with local models
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    det_model_dir="./models/det",
    rec_model_dir="./models/rec",
    cls_model_dir="./models/cls"
)

# Path to a valid .ttf font file (Modify this based on your OS)
FONT_PATH = "C:/Windows/Fonts/arial.ttf"  # For Windows
# FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # For Linux/macOS



UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    contents = await file.read()

  # Convert image to numpy array
    image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  # Perform OCR
    results = ocr.ocr(image, cls=True)


    extracted_text = []
    boxes = []
    for result in results:
        for line in result:
            boxes.append(line[0])  # Bounding box coordinates
            extracted_text.append(line[1][0])  # Extract detected text

    # Convert image to PIL format for drawing
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw OCR results with the specified font
    boxed_image = draw_ocr(image_pil, boxes, extracted_text, 
                           [line[1][1] for result in results for line in result], 
                           font_path=FONT_PATH)

    # Convert back to OpenCV format
    boxed_image = cv2.cvtColor(np.array(boxed_image), cv2.COLOR_RGB2BGR)

    # Save image with bounding boxes
    output_path = "D:\Prajwal\OCR Sample Images\output_with_boxes.jpg"
    cv2.imwrite(output_path, boxed_image)

    return {"text": extracted_text, "image_saved": output_path}