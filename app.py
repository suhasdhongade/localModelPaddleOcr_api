from fastapi import FastAPI, UploadFile, File, Response
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import os
from pdf2image import convert_from_path
import fitz  # PyMuPDF


app = FastAPI(
    title="OCR API with PaddleOCR",
    description="This API extracts text from images using PaddleOCR and provides annotated images with bounding boxes and text lists.",
    version="1.0",
    contact={
        "name": "Suhas Dhongade",
        "email": "email@example.com",
    }
)


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
#FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # For Linux/macOS

 
@app.post("/ocr/ExtractTextFromImage",summary="Extract text from image and return full text and lines with confidence score")
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


@app.post("/ocr/GetAnnotedImage", summary="Extract text from image and return annotated image")
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


@app.post("/ocr/", summary="Extract text from image and return full text and lines with confidence. Saves image at output path")
async def perform_ocr(file: UploadFile = File(...)):
    """
    This endpoint takes an image file as input, performs OCR to extract text,
    and returns the extracted text along with an annotated image.
    
    - **file**: Upload an image file.
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
    total_confidence = 0
    num_texts = 0
    line_items = []
    for idx, result in enumerate(results):
        for line in result:
            boxes.append(line[0])  # Bounding box coordinates
            extracted_text.append(line[1][0])  # Extract detected text
            confidences.append(line[1][1])
            full_text += line[1][0] + " "
            total_confidence += line[1][1]
            num_texts += 1
            line_items.append({"text": line[1][0], "confidence": line[1][1]})

    avg_confidence = total_confidence / num_texts if num_texts > 0 else 0

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
        draw.text((box[0][0], box[0][1] - 30), f"{i + 1}", fill="blue", font=font)
    
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

    # Save image with bounding boxes and numbers
    output_path = "D:/Prajwal/OCR Sample Images/output_with_numbers.jpg"
    cv2.imwrite(output_path, boxed_image)

    return {
        "full_text": full_text.strip(),
        "full_text_confidence": avg_confidence,
        "line_items": line_items,
        "image_saved": output_path
    }


@app.post("/ocr/ExtractTableFromImage", summary="Extract tables from an image")
async def extract_table(file: UploadFile = File(...)):
    """
    This endpoint takes an image file as input, performs OCR to extract tables,
    and returns the extracted table data in a structured format.
    
    - **file**: Upload an image file.
    - **Returns**: Extracted table data as a list of rows and columns.
    """
    contents = await file.read()

    # Convert image to numpy array
    try:
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Perform OCR
    results = ocr.ocr(image, cls=True)

    # Extract bounding boxes and text
    table_data = []
    for result in results:
        for line in result:
            box = line[0]  # Bounding box coordinates
            text = line[1][0]  # Detected text
            confidence = line[1][1]  # Confidence score
            table_data.append({"box": box, "text": text, "confidence": confidence})

    # Sort the data by the vertical position of the bounding boxes (top to bottom)
    table_data.sort(key=lambda x: (x["box"][0][1], x["box"][0][0]))

    # Group text into rows based on proximity of bounding boxes
    rows = []
    current_row = []
    row_threshold = 10  # Adjust this threshold based on your table layout
    last_y = None

    for item in table_data:
        y = item["box"][0][1]  # Top-left corner's y-coordinate
        if last_y is None or abs(y - last_y) < row_threshold:
            current_row.append(item["text"])
        else:
            rows.append(current_row)
            current_row = [item["text"]]
        last_y = y

    if current_row:
        rows.append(current_row)

    return {
        "table": rows
    }


@app.post("/convert_pdf_to_images/")
async def convert_pdf_to_images(file: UploadFile = File(...)):
    """Converts a PDF file into images and saves them to disk."""

    # Directory to save images
    OUTPUT_FOLDER = "saved_pages"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Read the uploaded PDF file as bytes
    pdf_bytes = await file.read()

    # Open PDF in memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    saved_images = []
    for i, page in enumerate(doc):
        # Render page as an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save image to disk
        img_path = os.path.join(OUTPUT_FOLDER, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        saved_images.append(img_path)

    return {"saved_images": saved_images}