from fastapi import APIRouter, UploadFile, File
import fitz  # PyMuPDF
from PIL import Image
import os

router = APIRouter()


@router.post("/pdf/extract_text_from_pdf/", summary="Extracts text from a electronically readable PDF file and returns it page-wise in JSON format")
async def extract_text_from_pdf(file: UploadFile = File(...)):
    """Extracts text from a PDF file and returns it page-wise in JSON format."""

    # Read the uploaded PDF file as bytes
    pdf_bytes = await file.read()

    # Open PDF in memory
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Extract text page-wise
    text_data = {f"page_{i+1}": page.get_text("text") for i, page in enumerate(doc)}

    return {"extracted_text": text_data}