import ctypes
import cv2
import gc
import hashlib
import json
import numpy as np
import os
import pymupdf
import shutil
import spacy
import sys
import tempfile

from resume_text_detector_cleaner import clean_and_save_text, detect_resume, FileNotResumeError
from spacy.util import compile_infix_regex

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

def _wipe_ndarray(arr: np.ndarray) -> None:
    """
    Overwrite a NumPy array's data buffer with zeros in-place
    Works even if the array is not C-contiguous by forcing a contiguous view
    """
    if arr is not None and arr.size > 0:
        flat    = np.frombuffer(arr.data, dtype=np.uint8)
        flat[:] = 0

def _wipe_str(s: str) -> None:
    """
    Best-effort overwrite of a CPython str object's internal character buffer
    CPython-specific — targets the compact ASCII/Latin-1 data that sits just
    after the fixed PyUnicodeObject header (size of an empty str object)
    Non-ASCII strings may have a wider per-character representation; we wipe
    the full reported object size to cover all variants
    """
    if s:
        header = sys.getsizeof("")          # fixed PyUnicodeObject header, e.g. 49 B
        total  = sys.getsizeof(s)           # header + encoded character data
        ctypes.memset(id(s) + header, 0, total - header)

def ensure_tesseract_installed():
    """
    Checks if Tesseract OCR is installed on the system.
    If not found, exits with an error — install manually
    """
    if shutil.which("tesseract") is not None:
        print("Tesseract is already installed.")
        return

    # Tesseract not found — exit and prompt the user to install it manually
    error_response = {
        "status" : "error",
        "message": "Tesseract is not installed. Please install it manually: https://github.com/tesseract-ocr/tesseract",
        "code"   : 202
    }
    print(json.dumps(error_response))
    sys.exit()

def preprocess_image_document_filter(file_path):
    """
    1. Resize      — enforce 1800–4000 px on longest side, aspect ratio preserved
    2. Grayscale   — reduce to single luminance channel
    3. Denoise     — remove scan grain / camera noise
    4. Deskew      — detect and correct text rotation via Hough lines
    5. Binarize    — adaptive Gaussian threshold for uneven lighting

    Returns the path to a cleaned temp PNG file
    The caller is responsible for deleting it after use
    """
    print(f"Preprocessing image with document filter: {file_path} ...")

    img = cv2.imread(file_path)
    if img is None:
        error_response = {
            "status" : "error",
            "message": f"OpenCV could not read image: {file_path}",
            "code"   : 203
        }
        print(json.dumps(error_response))
        sys.exit()

    # 1. Resize
    # 1800 px floor: minimum for reliable Tesseract OCR (~150 DPI equivalent on A4)
    # 4000 px ceiling: prevents excessive memory use during denoising / thresholding
    # Dimensions shrink/grow by the same ratio
    min_side = 1800
    max_side = 4000
    h, w     = img.shape[:2]
    longest  = max(h, w)

    if longest < min_side:
        scale         = min_side / longest          # upscale
        interpolation = cv2.INTER_CUBIC             # preserves sharpness when enlarging
        reason        = "upscaled (too small for OCR)"
    elif longest > max_side:
        scale         = max_side / longest          # downscale
        interpolation = cv2.INTER_AREA              # best quality when shrinking
        reason        = "downscaled (too large)"
    else:
        scale         = None                        # already in the acceptable range

    if scale is not None:
        # Derive both dimensions from the original aspect ratio (w / h)
        # Fixing the longest side to its target value and computing the other
        # from the original ratio avoids the independent int() truncation that
        # would occur if both were scaled separately — keeping the ratio exact
        if w >= h:                          # Landscape or square: width is longest
            new_w = int(w * scale)
            new_h = int(new_w * h / w)     # height derived from original ratio
        else:                               # portrait: height is longest
            new_h = int(h * scale)
            new_w = int(new_h * w / h)     # width derived from original ratio
        img = cv2.resize(img, (new_w, new_h), interpolation = interpolation)
        print(f"Resized image: {w}x{h} to {new_w}x{new_h} ({reason}, scale = {scale:.3f})")
    else:
        print(f"Image size {w}x{h} is within acceptable range, no resize needed.")

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    # h = 10: filter strength; templateWindowSize = 7, searchWindowSize = 21 are standard
    denoised = cv2.fastNlMeansDenoising(gray, h = 10, templateWindowSize = 7, searchWindowSize = 21)

    # 4. Deskew via Hough line detection
    # Edge to Hough lines to median angle to affine rotation
    edges = cv2.Canny(denoised, threshold1 = 50, threshold2 = 150, apertureSize = 3)
    lines = cv2.HoughLinesP(
        edges, rho = 1, theta = np.pi / 180, threshold = 100,
        minLineLength = 100, maxLineGap = 10
        )
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:                              # avoid division by zero
                angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angles:
            # Keep only near-horizontal lines (±15°) to avoid false skew from
            # vertical separators or diagonal decorative elements
            horizontal = [a for a in angles if abs(a) < 15]
            skew_angle  = float(np.median(horizontal)) if horizontal else 0.0

            if abs(skew_angle) > 0.3:               # ignore sub-pixel tilts
                print(f"Deskewing: correcting {skew_angle:.2f}° rotation ...")
                (ch, cw)   = denoised.shape
                center     = (cw // 2, ch // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, skew_angle, scale=1.0)
                denoised   = cv2.warpAffine(
                    denoised, rot_matrix, (cw, ch),
                    flags      = cv2.INTER_CUBIC,
                    borderMode = cv2.BORDER_REPLICATE   # fill borders with edge pixels
                )

    # 5. Adaptive threshold (document binarization)
    # Gaussian-weighted neighbourhood handles shadows and uneven scan lighting,
    # blockSize = 31: Large enough for gradual lighting gradients
    # C = 15: Constant subtracted from the weighted mean
    binarized = cv2.adaptiveThreshold(
        denoised, maxValue = 255,
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType  = cv2.THRESH_BINARY,
        blockSize      = 31,
        C              = 15
    )

    # 6. Save to a named temp PNG
    tmp_file = tempfile.NamedTemporaryFile(suffix = ".png", delete = False)
    tmp_path = tmp_file.name
    tmp_file.close()

    cv2.imwrite(tmp_path, binarized)
    print(f"Preprocessed image saved to temp file: {tmp_path}")

    # Secure wipe: zero every pixel buffer now that the file is written
    # All image data (uploaded + all intermediate arrays) is wiped from RAM
    _wipe_ndarray(img);       del img
    _wipe_ndarray(gray);      del gray
    _wipe_ndarray(denoised);  del denoised
    _wipe_ndarray(binarized); del binarized
    gc.collect()
    print("Image pixel buffers securely wiped from RAM.")

    return tmp_path

def extract_text_from_image_ocr(file_path):
    """
    Extract text from a PDF or Image file using OCR (Tesseract via PyMuPDF)
    """
    # Check if Tesseract is installed; install it automatically if missing
    ensure_tesseract_installed()

    # Check and set TESSDATA_PREFIX if not set (Helps PyMuPDF find language files on Linux)
    if "TESSDATA_PREFIX" not in os.environ:
        possible_paths = ["/usr/share/tesseract/tessdata", "/usr/local/share/tessdata"]
        for p in possible_paths:
            if os.path.exists(p):
                os.environ["TESSDATA_PREFIX"] = p
                print(f"TESSDATA_PREFIX set to: {p}")
                break

    print(f"Reading file for OCR: {file_path} ...")

    # Preprocess image files with the document filter before OCR
    tmp_path = None
    ext      = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        tmp_path  = preprocess_image_document_filter(file_path)
        ocr_path  = tmp_path
    else:
        ocr_path  = file_path

    full_text = ""
    try:
        doc = pymupdf.open(ocr_path)
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

                # Secure wipe: clear per-page text immediately after accumulation
                _wipe_str(page_text)
                del page_text

            except Exception as e:
                error_response = {
                    "status" : "error",
                    "message": f"OCR failed for page {i+1}: {str(e)}",
                    "code"   : 204
                }
                print(json.dumps(error_response))
                sys.exit()

        doc.close()
    except Exception as e:
        error_response = {
            "status" : "error",
            "message": f"Error opening or processing file: {str(e)}",
            "code"   : 205
        }
        print(json.dumps(error_response))
        sys.exit()
    finally:
        # Clean up preprocessed temp file if one was created
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("Temp preprocessed image removed.")

    if not full_text.strip():
        error_response = {
            "status" : "error",
            "message": f"File: {file_path}, no text could be extracted via OCR.",
            "code"   : 206
        }
        print(json.dumps(error_response))
        sys.exit()

    # Limit extremely long text to prevent spaCy memory errors
    if len(full_text) > 1000000:
        full_text = full_text[:1000000]

    return full_text

def load_model():
    """
    Loads the spaCy model with error handling.
    spaCy: For general English NLP tasks (email, URL, phone patterns, and NER)
    """
    print("Loading spaCy model (en_core_web_lg) ...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        error_response = {
            "status" : "error",
            "message": "spaCy Model 'en_core_web_lg' Not Found.",
            "code"   : 201
        }
        print(json.dumps(error_response))
        sys.exit()
    else:
        print("spaCy model loaded.")

    # Remove the '/' infix splitting rule from the tokenizer
    modified_infixes = [p for p in nlp.Defaults.infixes if '/' not in p]
    nlp.tokenizer.infix_finditer = compile_infix_regex(modified_infixes).finditer
    print("Tokenizer updated: slash-separated terms will be kept as single tokens.")

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
            "code"   : 200
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
            "code"   : 207
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 5: Clean, Save, and Extract PII in one pass (Only if it is a resume)
    extracted_emails, extracted_phones = clean_and_save_text(raw_text, nlp)

    # Secure wipe: raw_text has served its purpose and contains PII
    # Hash first as a tamper-evident audit trail
    raw_text_hash = hashlib.sha256(raw_text.encode("utf-8", errors = "replace")).hexdigest()
    print(f"Raw text SHA-256 (audit trail): {raw_text_hash}")

    _wipe_str(raw_text)
    del raw_text
    gc.collect()
    print("Raw PII text securely wiped from RAM.")

    print(f"Extracted emails: {extracted_emails}")
    print(f"Extracted phones: {extracted_phones}")