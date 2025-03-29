from fastapi import FastAPI, UploadFile, File
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
import numpy as np
import cv2
import io

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
    confidences = []
    for idx, result in enumerate(results):
        for line in result:
            boxes.append(line[0])  # Bounding box coordinates
            extracted_text.append(line[1][0])  # Extract detected text
            confidences.append(line[1][1])

    # Convert image to PIL format for drawing
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_width, img_height = image_pil.size

    # Extend the image to the right to add text list
    extended_width = img_width + 300  # Add space for text list
    extended_image = Image.new("RGB", (extended_width, img_height), "white")
    extended_image.paste(image_pil, (0, 0))

    draw = ImageDraw.Draw(extended_image)
    font = ImageFont.truetype(FONT_PATH, 25)
    
    # Draw bounding boxes and numbers
    for i, box in enumerate(boxes):
        box = [(int(x), int(y)) for x, y in box]  # Convert to integer tuples
        draw.polygon(box, outline="red", width=2)
        draw.text((box[0][0], box[0][1] - 30), f"{i + 1}", fill="red", font=font)
    
    # Add extracted text list on the right side of the image
    text_x, text_y = img_width + 10, 10
    for i, (text, confidence) in enumerate(zip(extracted_text, confidences)):
        draw.text((text_x, text_y + i * 30), f"{i + 1}: {text} ({confidence:.3f})", fill="black", font=font)
    
    # Convert back to OpenCV format
    boxed_image = cv2.cvtColor(np.array(extended_image), cv2.COLOR_RGB2BGR)

    # Save image with bounding boxes and numbers
    output_path = "D:/Prajwal/OCR Sample Images/output_with_numbers.jpg"
    cv2.imwrite(output_path, boxed_image)

    return {"text": extracted_text, "image_saved": output_path}
