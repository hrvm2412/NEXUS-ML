# AI
import json
import os
import pymupdf
import sys
import spacy

from resume_text_detector_cleaner import clean_and_save_text, detect_resume, FileNotResumeError

def extract_text_from_image_ocr(file_path):
    """
    Extract text from a PDF or Image file using OCR (Tesseract via PyMuPDF).
    """
    # Check and set TESSDATA_PREFIX if not set (Helps PyMuPDF find language files on Linux)
    if "TESSDATA_PREFIX" not in os.environ:
        possible_paths = ["/usr/share/tesseract/tessdata", "/usr/local/share/tessdata"]
        for p in possible_paths:
            if os.path.exists(p):
                os.environ["TESSDATA_PREFIX"] = p
                print(f"TESSDATA_PREFIX set to: {p}")
                break

    print(f"Reading file for OCR: {file_path} ...")

    full_text = ""
    try:
        doc = pymupdf.open(file_path)
        print("Performing OCR extraction (this requires Tesseract installed) ...")

        for i, page in enumerate(doc):
            try:
                # Create a TextPage with OCR enabled
                # flags = 3: Preserves ligatures and whitespace
                # dpi = 300: Standard resolution for OCR
                # full = True: Scans the entire page content as an image
                textpage = page.get_textpage_ocr(flags = 3, language='eng', dpi = 300, full = True)
                
                # Extract text from the OCR'd textpage
                # "text" mode preserves visual line breaks as '\n' characters
                page_text  = page.get_text("text", textpage = textpage)
                full_text += page_text + "\n"
                
                line_count = len(page_text.splitlines())
                print(f"Processed page {i+1}/{len(doc)}: Extracted {line_count} lines (formatting preserved)")
            except Exception as e:
                print(f"OCR failed for page {i+1}: {e}")

        doc.close()
    except Exception as e:
        print(f"Error opening or processing file: {e}")

    if not full_text.strip():
        error_response = {
            "status" : "error",
            "message": f"File: {file_path}, no text could be extracted via OCR.",
            "code"   : 503
        }
        print(json.dumps(error_response))
        sys.exit()

    # Limit extremely long text to prevent spaCy memory errors
    if len(full_text) > 1000000:
        full_text = full_text[:1000000]

    return full_text

def load_model():
    """
    Loads the spaCy model with error handling
    """
    print("Loading spaCy model ...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        error_response = {
            "status" : "error",
            "message": "spaCy Model 'en_core_web_lg' Not Found.",
            "code"   : 500
        }
        print(json.dumps(error_response))
        sys.exit()
    else:
        print("Model loaded.")
        return nlp

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input path (Targeting an image-based resume)
    input_filename = "Resumes/Resume_own_image.png"
    input_path = os.path.join(script_dir, input_filename)

    # STEP 1: Check File Existence
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Resume file not found: {input_path}",
            "code"   : 502
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 2: Load Model
    nlp = load_model()

    # STEP 3: Extract Text (OCR)
    raw_text = extract_text_from_image_ocr(input_path)

    # STEP 4: Detect Resume
    try:
        if detect_resume(raw_text, nlp):
            print("This file is a resume.")
        else:
            raise FileNotResumeError
    except FileNotResumeError:
        error_response = {
            "status" : "error",
            "message": "This file is NOT a resume.",
            "code"   : 400
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 5: Clean and Save (Only if it is a resume)
    clean_and_save_text(raw_text, nlp)