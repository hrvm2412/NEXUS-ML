import csv
import ctypes
import gc
import hashlib
import json
import os
import pymupdf
import spacy
import sys

from spacy.matcher import Matcher, PhraseMatcher
from spacy.util import compile_infix_regex

class FileExceedsModelLimitError(Exception):
    pass

class FileExceedsPageLimitError(Exception):
    pass

class FileNotResumeError(Exception):
    pass

class LineExceedsModelLimitError(Exception):
    pass

def _wipe_str(s: str) -> None:
    """
    Non-ASCII strings may have a wider per-character representation; we wipe
    the full reported object size to cover all variants
    """
    if s:
        header = sys.getsizeof("")          # fixed PyUnicodeObject header, e.g. 49 B
        total  = sys.getsizeof(s)           # header + encoded character data
        ctypes.memset(id(s) + header, 0, total - header)

def clean_and_save_text(text, nlp_spacy):
    """
    Removes PII and saves the cleaned text
    spaCy: Detects names (PERSON), email, URL, and phone patterns
    CSVs: Locations (barangay, city, province, region) and zipcodes

    Preprocessing methods per detection type:
    - Phone numbers   : No preprocessing
    - Email / URL     : No preprocessing
    - Phone prefixes  : No preprocessing
    - Names (PERSON)  : No preprocessing
    - Locations/Zip   : Exact whole-row match (case-insensitive) via PhraseMatcher
                        Parentheses are treated as part of the location name

    Also extracts emails and phone numbers (normalized to 09XXXXXXXXX)
    as variables alongside redaction, reusing the same matcher pass
    """

    print("Preprocessing text (Cleaning & PII Removal) ...")

    phil_loc_path      = os.path.join(os.path.dirname(__file__), "Phil_loc")
    barangay_locations = load_locations_from_csv(os.path.join(phil_loc_path, "barangays.csv"))
    city_locations     = load_locations_from_csv(os.path.join(phil_loc_path, "cities.csv"))
    province_locations = load_locations_from_csv(os.path.join(phil_loc_path, "provinces.csv"))
    region_locations   = load_locations_from_csv(os.path.join(phil_loc_path, "regions.csv"))
    zipcode_locations  = load_locations_from_csv(os.path.join(phil_loc_path, "zipcodes_only.csv"))
    
    locations = barangay_locations | city_locations | province_locations | region_locations | zipcode_locations
    locations.add("philippines")

    # PhraseMatcher for exact whole-row location matching (case-insensitive)
    # Parentheses are preserved as part of the phrase (e.g., "Adams (Pob.)", "Region I (Ilocos Region)")
    phrase_matcher = PhraseMatcher(nlp_spacy.vocab, attr="LOWER")
    phrase_matcher.add("LOCATION", [nlp_spacy.make_doc(loc) for loc in locations])

    # Matcher for Philippine phone number patterns
    matcher = Matcher(nlp_spacy.vocab)

    matcher.add("PHONE", get_phone_patterns())

    # PII extraction accumulators
    extracted_emails = []
    extracted_phones = []

    lines          = text.split('\n')
    filtered_lines = []

    for line in lines:
        try:
            if len(line) > nlp_spacy.max_length:
                raise LineExceedsModelLimitError
        except LineExceedsModelLimitError:
            error_response = {
                "status" : "error",
                "message": f"A line of text exceeds the spaCy model (en_core_web_lg) token limit and cannot be processed: '{line[:50]}'",
                "code"   : 106
            }
            print(json.dumps(error_response))
            sys.exit()

        doc_spacy      = nlp_spacy(line)
        redact_indices = set()

        # 1. Matcher PII (Phone patterns) - No preprocessing
        #    Reused for both redaction and extraction in the same pass
        matches = matcher(doc_spacy)
        for match_id, start, end in matches:
            for i in range(start, end):
                redact_indices.add(i)

            # Extract: normalize matched span and collect
            raw_phone  = doc_spacy[start:end].text
            normalized = normalize_phone(raw_phone)
            if normalized and normalized not in extracted_phones:
                extracted_phones.append(normalized)

        # 2. Token PII from spaCy (email, URL, phone prefixes, names, locations, and zipcodes)
        for i, token in enumerate(doc_spacy):
            
            # Emails and URLs
            # Preprocessing: None
            if token.like_email or token.like_url:
                redact_indices.add(i)

                # Extract: collect email (URLs are redacted but not collected)
                if token.like_email:
                    email = token.text.strip()
                    if email and email not in extracted_emails:
                        extracted_emails.append(email)

            # Manual phone prefix check
            # Preprocessing: None
            elif token.text.startswith('+') and any(c.isdigit() for c in token.text):
                redact_indices.add(i)

            # Name detection (PERSON)
            # Preprocessing: None
            elif token.ent_type_ == "PERSON":
                redact_indices.add(i)

        # 3. Location and zipcode detection via PhraseMatcher
        #    Exact whole-row match (case-insensitive); parentheses are part of the phrase
        loc_matches = phrase_matcher(doc_spacy)
        for _, start, end in loc_matches:
            for i in range(start, end):
                redact_indices.add(i)
        
        # 4. Remove phone-related punctuation around redacted tokens (parentheses, dashes, dots)
        # This handles cases like "(+63)" or "918-744-2414" where punctuation is separate tokens
        for i, token in enumerate(doc_spacy):
            if i not in redact_indices and token.text in ['(', ')', '-', '.']:
                # If surrounded by redacted tokens, redact the punctuation too
                has_redacted_before = i > 0 and (i - 1) in redact_indices
                has_redacted_after  = i < len(doc_spacy) - 1 and (i + 1) in redact_indices
                if has_redacted_before or has_redacted_after:
                    redact_indices.add(i)
        
        # Reconstruct - only add non-redacted tokens
        reconstructed_line = ""
        for i, token in enumerate(doc_spacy):
            if i not in redact_indices:
                reconstructed_line += token.text + token.whitespace_
        
        # Clean up extra whitespace (handles multiple spaces from redaction)
        reconstructed_line = " ".join(reconstructed_line.split())
        filtered_lines.append(reconstructed_line)

    # Join with newlines to preserve structure
    cleaned_text_with_structure = "\n".join(filtered_lines)
    
    # Tokenize and lowercase for final output

    doc = nlp_spacy(cleaned_text_with_structure)
    cleaned_tokens = [token.text.lower() for token in doc if not token.is_space]
    cleaned_text = " ".join(cleaned_tokens)

    # Apply smart casing preprocessing for extraction
    preprocessed_text = preprocess_for_extraction(cleaned_text_with_structure, nlp_spacy)

    # Save both versions
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Tokenized and lowercased
    output_path_cleaned = os.path.join(output_dir, "Resume_text/Resume_cleaned.txt")
    with open(output_path_cleaned, "w", encoding = "utf-8") as f:
        f.write(cleaned_text)

    # 2. Preprocessed for extraction
    output_path_for_extraction = os.path.join(output_dir, "Resume_text/Resume_for_extraction.txt")
    with open(output_path_for_extraction, "w", encoding = "utf-8") as f:
        f.write(preprocessed_text)
        
    print(f"Cleaned text (tokenized/lowercased) saved to: {output_path_cleaned}")
    print(f"Preprocessed text (smart casing) saved to: {output_path_for_extraction}")

    # Secure wipe: all intermediate text derived from PII-bearing input
    # Files are written; the in-memory strings no longer serve any purpose
    # Overwrite their character buffers before releasing references so that GC
    # cannot expose the content via a dangling object in the heap
    _wipe_str(cleaned_text_with_structure); del cleaned_text_with_structure
    _wipe_str(cleaned_text);               del cleaned_text
    _wipe_str(preprocessed_text);          del preprocessed_text
    gc.collect()
    print("Intermediate PII-derived text strings securely wiped from RAM.")

    return extracted_emails, extracted_phones

def detect_resume(text, nlp):
    """
    Analyzes raw text to determine if it is a resume

    Preprocessing methods per detection type:
    - Phone numbers        : No preprocessing
    - Email detection      : No preprocessing
    - Headers              : No preprocessing
    - Entity density       : No preprocessing
    - Location/Zip density : Exact whole-row match (case-insensitive) via PhraseMatcher
                             Parentheses are treated as part of the location name
    """

    print("Analyzing text for resume detection ...")

    try:
        if len(text) > 1000000:
            raise FileExceedsModelLimitError
    except FileExceedsModelLimitError:
        error_response = {
            "status" : "error",
            "message": "Resume text exceeds spaCy model limit.",
            "code"   : 104
        }
        print(json.dumps(error_response))
        sys.exit()
        
    # Load CSV locations for entity density detection
    phil_loc_path      = os.path.join(os.path.dirname(__file__), "Phil_loc")
    barangay_locations = load_locations_from_csv(os.path.join(phil_loc_path, "barangays.csv"))
    city_locations     = load_locations_from_csv(os.path.join(phil_loc_path, "cities.csv"))
    province_locations = load_locations_from_csv(os.path.join(phil_loc_path, "provinces.csv"))
    region_locations   = load_locations_from_csv(os.path.join(phil_loc_path, "regions.csv"))
    zipcode_locations  = load_locations_from_csv(os.path.join(phil_loc_path, "zipcodes_only.csv"))
    
    locations = barangay_locations | city_locations | province_locations | region_locations | zipcode_locations
    locations.add("philippines")
    
    # PhraseMatcher for exact whole-row location matching (case-insensitive)
    # Parentheses are preserved as part of the phrase (e.g., "Adams (Pob.)", "Region I (Ilocos Region)")
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    phrase_matcher.add("LOCATION", [nlp.make_doc(loc) for loc in locations])

    doc = nlp(text)

    # STEP 1: Contact Info - No preprocessing
    matcher = Matcher(nlp.vocab)
    
    # Register all phone patterns with the matcher under the label "PHONE"
    matcher.add("PHONE", get_phone_patterns())
    
    has_phone = len(matcher(doc)) > 0

    # STEP 2: Email Detection - No preprocessing
    has_email = any(token.like_email for token in doc)
    
    # STEP 3: Headers - No preprocessing
    header_matcher  = Matcher(nlp.vocab)
    header_patterns = [
        [{"LOWER": {"IN": ["education", "academics", "qualifications"]}}],
        [{"LOWER": "academic"}, {"LOWER": {"IN": ["background", "history"]}}],
        [{"LOWER": {"IN": ["experience", "employment", "career"]}}],
        [{"LOWER": {"IN": ["work", "professional"]}}, {"LOWER": {"IN": ["history", "experience", "background"]}}],
        [{"LOWER": {"IN": ["skills", "technologies", "competencies", "expertise"]}}],
        [{"LOWER": {"IN": ["technical", "core", "hard", "soft"]}}, {"LOWER": "skills"}],
        [{"LOWER": {"IN": ["summary", "objective", "profile"]}}],
        [{"LOWER": "about"}, {"LOWER": "me"}],
        [{"LOWER": {"IN": ["projects", "portfolio", "certifications", "licenses", "awards", "achievements"]}}],
        [{"LOWER": {"IN": ["references", "languages", "publications", "interests"]}}],
        [{"LOWER": {"IN": ["volunteering", "volunteer", "volunteer experience", "community service"]}}],
        [{"LOWER": {"IN": ["memberships", "professional memberships", "associations", "affiliations"]}}],
        [{"LOWER": {"IN": ["training", "workshops", "professional development", "courses"]}}],
        [{"LOWER": {"IN": ["patents", "publications", "research", "intellectual property"]}}],
        [{"LOWER": {"IN": ["honors", "distinctions", "recognitions", "scholarships"]}}],
        [{"LOWER": {"IN": ["internship", "internships", "practicum"]}}],
        [{"LOWER": {"IN": ["core competencies", "key competencies", "core strengths"]}}],
        [{"LOWER": {"IN": ["strengths", "personal strengths", "key strengths"]}}]]
    header_matcher.add("RESUME_HEADER", header_patterns)
    
    # Search the text for all resume section headers
    header_matches = header_matcher(doc)
    
    # Create a set to store the header keywords we find (without repeats)
    found_keywords = set()
    
    # Go through each match and save the header text
    for _, start, end in header_matches:
        found_keywords.add(doc[start:end].text.lower())
    
    # Count how many different headers we found
    keyword_count = len(found_keywords)

    # STEP 4: Entity Density - Only locations and zipcodes are preprocessed
    entity_count    = 0
    relevant_labels = ["DATE", "ORG", "PERSON"]
    
    # Count spaCy entities (DATE, ORG, PERSON)
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            entity_count += 1
    
    # Count CSV-based location and zipcode entities via exact PhraseMatcher
    loc_matches = phrase_matcher(doc)
    entity_count += len(loc_matches)

    # Scoring
    score = 0
    if has_email: score += 2
    if has_phone: score += 2
    score += min(keyword_count, 5)
    if entity_count > 5: score += 1

    threshold = 4
    
    return score >= threshold

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF file
    """
    print(f"Reading resume PDF from: {pdf_path} ...")

    try:
        doc  = pymupdf.open(pdf_path)
        num_pages = doc.page_count

        if num_pages > 2:
            raise FileExceedsPageLimitError

        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        
        if not text.strip():
            raise ValueError
    except FileExceedsPageLimitError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF: {pdf_path}, exceeds page limit (2 pages maximum).",
            "code"   : 102
        }
        print(json.dumps(error_response))
        sys.exit()
    except ValueError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF: {pdf_path}, no text could be extracted.",
            "code"   : 103
        }
        print(json.dumps(error_response))
        sys.exit()

    print("Text extracted from PDF.")

    return text

def get_phone_patterns():
    """Returns the shared list of Philippine phone number patterns for Matcher"""
    sep      = {"ORTH": {"IN": ["-", " "]}, "OP": "?"}
    digit_1  = {"IS_DIGIT": True, "LENGTH": 1}
    digit_2  = {"IS_DIGIT": True, "LENGTH": 2}
    digit_3  = {"IS_DIGIT": True, "LENGTH": 3}
    digit_4  = {"IS_DIGIT": True, "LENGTH": 4}
    digit_9  = {"IS_DIGIT": True, "LENGTH": 9}
    digit_11 = {"IS_DIGIT": True, "LENGTH": 11}
    
    return [
        # Format: 09XXXXXXXXX (11 digits as single token)
        [digit_11],
        
        # Format: 09XX-XXX-XXXX or 09XX XXX XXXX
        [{"ORTH": "0"}, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Format: 0XXX-XXX-XXXX (e.g., 0917-744-2414 where 0917 is one token)
        [digit_4, sep, digit_3, sep, digit_4],

        # Format: 09-XXXXXXXXX or 09 XXXXXXXXX
        [{"ORTH": "0"}, digit_1, sep, digit_9],
        
        # Format: +639XXXXXXXXX
        [{"ORTH": "+63"}, digit_1, digit_9],
        
        # Format: +63-9XX-XXX-XXXX or +63 9XX XXX XXXX (digit_1 + digit_2 variant)
        [{"ORTH": "+63"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Format: +63-9XX-XXX-XXXX or +63 9XX-XXX-XXXX (3-digit prefix variant)
        [{"ORTH": "+63"}, sep, digit_3, sep, digit_3, sep, digit_4],
        
        # Format: +63-9XXXXXXXXX or +63 9XXXXXXXXX
        [{"ORTH": "+63"}, sep, digit_1, digit_9],
        
        # Format: 639XXXXXXXXX
        [{"ORTH": "63"}, digit_1, digit_9],
        
        # Format: 63-9XX-XXX-XXXX or 63 9XX XXX XXXX (digit_1 + digit_2 variant)
        [{"ORTH": "63"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Format: 63-9XX-XXX-XXXX or 63 9XX-XXX-XXXX (3-digit prefix variant)
        [{"ORTH": "63"}, sep, digit_3, sep, digit_3, sep, digit_4],
        
        # Format: 63-9XXXXXXXXX or 63 9XXXXXXXXX
        [{"ORTH": "63"}, sep, digit_1, digit_9],
        
        # Format: (+63)9XXXXXXXXX or (+63)-9XXXXXXXXX or (+63) 9XXXXXXXXX
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_1, digit_9],
        
        # Format: (+63)9XX-XXX-XXXX or (+63)-9XX-XXX-XXXX (digit_1 + digit_2 variant)
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Format: (+63) 9XX-XXX-XXXX where 9XX is a single 3-digit token
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_3, sep, digit_3, sep, digit_4],
    ]

def load_locations_from_csv(file_path):
    """
    Loads locations and zipcodes from CSV file as exact whole-row strings.
    Parentheses are preserved as part of the location name
    (e.g., "Adams (Pob.)", "Region I (Ilocos Region)")
    Works for barangays.csv, cities.csv, provinces.csv, regions.csv, and zipcodes_only.csv
    """
    locations = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if not row: continue
            text = row[0].strip()
            if text:
                locations.add(text.lower())
    return locations

def load_model():
    """
    Loads the spaCy model
    spaCy: For general English NLP tasks (email, URL, phone patterns, and NER)
    """

    print("Loading spaCy model (en_core_web_lg) ...")

    try:
        nlp_spacy = spacy.load("en_core_web_lg")
    except OSError:
        error_response = {
            "status" : "error",
            "message": "spaCy model 'en_core_web_lg' not found.",
            "code"   : 101
        }
        print(json.dumps(error_response))
        sys.exit()
    else:
        print("spaCy model loaded.")

    # FIX: Remove the '/' infix splitting rule from the tokenizer
    modified_infixes = [p for p in nlp_spacy.Defaults.infixes if '/' not in p]
    nlp_spacy.tokenizer.infix_finditer = compile_infix_regex(modified_infixes).finditer

    print("Tokenizer updated: Slash-separated terms will be kept as single tokens.")
    
    return nlp_spacy

def normalize_phone(raw_phone):
    """
    Normalizes a Philippine phone number (in any supported format) to 09XXXXXXXXX

    Strips all non-digit characters from the matched span, then converts:
      - 639XXXXXXXXX  (12 digits, country code without +) to 09XXXXXXXXX
      - 9XXXXXXXXX    (10 digits, no leading 0)           to 09XXXXXXXXX
      - 09XXXXXXXXX   (11 digits, already local format)   to unchanged

    Returns the normalized 11-digit string, or None if the result is invalid
    """
    digits = ''.join(c for c in raw_phone if c.isdigit())

    if digits.startswith("63") and len(digits) == 12:
        digits = "0" + digits[2:]
    elif digits.startswith("9") and len(digits) == 10:
        digits = "0" + digits

    if len(digits) == 11 and digits.startswith("09"):
        return digits
    return None

def preprocess_for_extraction(text, nlp):
    """
    Preprocess text for skill/knowledge extraction using context-aware casing:
    - Preserves Acronyms (e.g. SQL, AWS, C/C++)
    - Preserves Proper Nouns (e.g. Python, Docker, Google)
    - Lowercases common words (verbs, adjectives), even at the start of sentences
    """
    print("Preprocessing text for extraction (Smart Casing) ...")
    
    lines = text.split('\n')
    processed_lines = []
    
    # Categories to preserve original casing
    preserve_ent_types = ["ORG", "PRODUCT", "GPE", "PERSON", "NORP", "LANGUAGE"]

    for line in lines:
        doc = nlp(line)
        processed = ""
        
        for token in doc:
            # RULE 1: Acronyms
            if token.text.isupper() and len(token.text) > 1:
                processed += token.text + token.whitespace_
            
            # RULE 2: Proper Nouns
            elif token.pos_ == "PROPN" or token.ent_type_ in preserve_ent_types:
                processed += token.text + token.whitespace_
                
            # RULE 3: Common Words
            else:
                processed += token.text.lower() + token.whitespace_
        
        processed_lines.append(processed.strip())

    return '\n'.join(processed_lines)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path   = os.path.join(script_dir, "Resumes/Resume_own.pdf")

    # STEP 1: Check File Existence
    print("Checking if resume PDF exists ...")
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF not found: {pdf_path}",
            "code"   : 100
        }
        print(json.dumps(error_response))
        sys.exit()
    print("Resume PDF found.")

    # STEP 2: Load Model
    nlp_spacy = load_model()

    # STEP 3: Extract Text
    raw_text = extract_text_from_pdf(pdf_path)

    # STEP 4: Detect Resume
    try:
        if detect_resume(raw_text, nlp_spacy):
            print("This file is a resume.")
        else:
            raise FileNotResumeError
    except FileNotResumeError:
        error_response = {
            "status" : "error",
            "message": "This file is NOT a resume.",
            "code"   : 105
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 5: Clean, Save, and Extract PII in one pass
    extracted_emails, extracted_phones = clean_and_save_text(raw_text, nlp_spacy)

    # Secure wipe: raw_text has served its purpose and contains PII
    # Hash first as a tamper-evident audit trail (SHA-256 is one-way; no PII
    # can be reconstructed from the digest)
    raw_text_hash = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
    print(f"Raw text SHA-256 (audit trail): {raw_text_hash}")

    _wipe_str(raw_text)
    del raw_text
    gc.collect()
    print("Raw PII text securely wiped from RAM.")

    print(f"Extracted emails: {extracted_emails}")
    print(f"Extracted phones: {extracted_phones}")