from fastapi import FastAPI
from endpoints import   ocr_annotated_image, ocr_extract_text,ocr_extract_text_and_AnnotateImage, ocr_table_extract, pdf_to_images, pdf_to_text

app = FastAPI(
    title="OCR API with PaddleOCR",
    description="This API extracts text from images using PaddleOCR and provides annotated images with bounding boxes and text lists.",
    version="1.0",
    contact={
        "name": "Suhas Dhongade",
        "email": "email@example.com",
    }
)

# Include routers from modular files
app.include_router(ocr_extract_text.router)
app.include_router(ocr_annotated_image.router)
app.include_router(ocr_extract_text_and_AnnotateImage.router)
app.include_router(ocr_table_extract.router)

# PDF related endpoints
app.include_router(pdf_to_text.router)
app.include_router(pdf_to_images.router)

