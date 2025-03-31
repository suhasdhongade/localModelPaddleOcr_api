from fastapi import APIRouter, UploadFile, File
import fitz  # PyMuPDF
from PIL import Image
import os

router = APIRouter()

@router.post("/pdf/convert_pdf_to_images/", summary="Convert PDF to images")
async def convert_pdf_to_images(file: UploadFile = File(...)):
    """
    Convert a PDF file into images and save them to disk.
    """
    OUTPUT_FOLDER = "saved_pages"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    saved_images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_path = os.path.join(OUTPUT_FOLDER, f"page_{i + 1}.png")
        img.save(img_path, "PNG")
        saved_images.append(img_path)

    return {"saved_images": saved_images}