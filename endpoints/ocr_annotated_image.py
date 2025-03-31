from fastapi import APIRouter, UploadFile, File, Response
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
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

FONT_PATH = "C:/Windows/Fonts/arial.ttf"  # Update this path as needed

@router.post("/ocr/get_annoted_image", summary="Extract text from image and return annotated image")
async def get_annoted_image(file: UploadFile = File(...)):
    """
    This endpoint takes an image file as input, performs OCR and retruns image with bounding boxes and numbers.
    
    - **file**: Upload an image file.
    - **Returns**: Extracted text, confidence scores.
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
    boxes = []
    confidences = []
    full_text = ""
    line_items = []
    for idx, result in enumerate(results):
        for line in result:
            boxes.append(line[0])  # Bounding box coordinates
            extracted_text.append(line[1][0])  # Extract detected text
            confidences.append(line[1][1])
            full_text += line[1][0] + " "
            line_items.append({"text": line[1][0], "confidence": line[1][1]})

    # Convert image to PIL format for drawing
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_width, img_height = image_pil.size

    # Extend the image to the right to add text list
    extended_width = img_width + img_width  # Add space for text list
    extended_image = Image.new("RGB", (extended_width, img_height), "white")
    extended_image.paste(image_pil, (0, 0))

    draw = ImageDraw.Draw(extended_image)
    font = ImageFont.truetype(FONT_PATH, 12)
    
    # Draw bounding boxes and numbers
    for i, box in enumerate(boxes):
        box = [(int(x), int(y)) for x, y in box]  # Convert to integer tuples
        draw.polygon(box, outline="red", width=2)
        draw.text((box[0][0], box[0][1] - 10), f"{i + 1}", fill="blue", font=font)
    
    # Add extracted text list on the right side of the image
    text_x, text_y = img_width + 10, 10
    max_height = img_height - 10  # Ensure text does not overflow
    line_spacing = 15  # Space between lines
    
    for i, (text, confidence) in enumerate(zip(extracted_text, confidences)):
        if text_y + line_spacing > max_height:
            text_y = 10  # Reset to top if overflow occurs
            text_x += 300  # Move to the right for additional column
        draw.text((text_x, text_y), f"{i + 1}: {text} ({confidence:.3f})", fill="black", font=font)
        text_y += line_spacing
    
    # Convert back to OpenCV format
    boxed_image = cv2.cvtColor(np.array(extended_image), cv2.COLOR_RGB2BGR)

    _, img_encoded = cv2.imencode(".jpg", boxed_image)
    
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
