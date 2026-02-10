# AI
import json
import os
import pymupdf
import sys
import spacy

from spacy.matcher import Matcher

class FileNotResumeError(Exception):
    pass

def preprocess_for_extraction(text, nlp):
    """
    Preprocess text for skill/knowledge extraction using Context-Aware Casing:
    - Preserves Acronyms (e.g., SQL, AWS).
    - Preserves Proper Nouns (e.g., Python, Docker, Google).
    - Lowercases common words (verbs, adjectives), even at the start of sentences.
    """
    print("Preprocessing text for extraction (Smart Casing)...")
    
    lines = text.split('\n')
    processed_lines = []
    
    # Categories to preserve original casing
    preserve_ent_types = ["ORG", "PRODUCT", "GPE", "PERSON", "NORP", "LANGUAGE"]

    for line in lines:
        doc = nlp(line)
        processed = ""
        
        for token in doc:
            # RULE 1: Acronyms (Keep Uppercase)
            # e.g., "API", "ML", "AWS"
            if token.text.isupper() and len(token.text) > 1:
                processed += token.text + token.whitespace_
            
            # RULE 2: Proper Nouns (Keep Title Case)
            # e.g., "Python", "JavaScript", "React"
            # We check if it is a Proper Noun (PROPN) OR falls into specific Entity types
            elif token.pos_ == "PROPN" or token.ent_type_ in preserve_ent_types:
                processed += token.text + token.whitespace_
                
            # RULE 3: Common Words (Lowercase)
            # This handles standardizing bullet points: "Developed" -> "developed"
            else:
                processed += token.text.lower() + token.whitespace_
        
        processed_lines.append(processed.strip())

    return '\n'.join(processed_lines)

def clean_and_save_text(text, nlp):
    """
    Removes PII and saves the cleaned text
    """
    print("Preprocessing text (Cleaning & PII Removal) ...")
    
    # Re-use patterns for cleaning
    matcher = Matcher(nlp.vocab)
    # Same patterns as detect_resume code, redefined here for scope
    d_3   = {"IS_DIGIT": True, "LENGTH": 3}
    d_4   = {"IS_DIGIT": True, "LENGTH": 4}
    d_var = {"IS_DIGIT": True, "LENGTH": {">=": 1, "<=": 4}}
    sep   = {"ORTH": {"IN": ["-", ".", "/"]}, "OP": "?"}

    phone_patterns = [
        [{"ORTH": "("}, d_3, {"ORTH": ")"}, sep, d_3, sep, d_4],
        [d_3, sep, d_3, sep, d_4],
        [{"ORTH": "+"}, d_var, sep, d_var, sep, d_var, sep, {"IS_DIGIT": True, "LENGTH": {">=": 3, "<=": 9}}],
        [{"ORTH": "+"}, d_var, sep, {"ORTH": "("}, d_3, {"ORTH": ")"}, sep, d_3, sep, d_4],
        [{"IS_DIGIT": True, "LENGTH": {">=": 10, "<=": 15}}]
    ]
    matcher.add("PHONE", phone_patterns)

    lines          = text.split('\n')
    filtered_lines = []

    for line in lines:
        doc_line = nlp(line)
        redact_indices = set()

        # Matcher PII
        matches = matcher(doc_line)
        for match_id, start, end in matches:
            for i in range(start, end):
                redact_indices.add(i)

        # Token PII
        for i, token in enumerate(doc_line):
            if token.ent_type_ in ["PERSON", "GPE", "LOC", "FAC"]:
                redact_indices.add(i)
            elif token.like_email or token.like_url:
                redact_indices.add(i)
            elif token.text.startswith('+') and any(c.isdigit() for c in token.text):
                redact_indices.add(i)
        
        # Reconstruct
        reconstructed_line = ""
        for i, token in enumerate(doc_line):
            if i in redact_indices:
                reconstructed_line += "" + token.whitespace_
            else:
                reconstructed_line += token.text + token.whitespace_
        
        filtered_lines.append(reconstructed_line.strip())

    # Join with newlines to preserve structure
    cleaned_text_with_structure = "\n".join(filtered_lines)
    
    # Tokenize and Lowercase for final output (Restored behavior for Resume_preprocessed.txt)
    print("Tokenizing and lowercasing ...")
    doc = nlp(cleaned_text_with_structure)
    cleaned_tokens = [token.text.lower() for token in doc if not token.is_space]
    cleaned_text = " ".join(cleaned_tokens)

    # Apply Smart Casing preprocessing for extraction
    preprocessed_text = preprocess_for_extraction(cleaned_text_with_structure, nlp)

    # Save both versions
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Tokenized and Lowercased (Restored behavior)
    output_path_original = os.path.join(output_dir, "Resume_preprocessed.txt")
    with open(output_path_original, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    # 2. Preprocessed for extraction (Smart Casing applied)
    output_path_preprocessed = os.path.join(output_dir, "Resume_preprocessed_no_space.txt")
    with open(output_path_preprocessed, "w", encoding="utf-8") as f:
        f.write(preprocessed_text)
        
    print(f"Cleaned text (tokenized/lowercased) saved to: {output_path_original}")
    print(f"Preprocessed text (Smart Casing) saved to: {output_path_preprocessed}")

def detect_resume(text, nlp):
    """
    Analyzes raw text to determine if it is a resume
    """
    print("Analyzing text for resume detection ...")
    
    if text is None:
        return False

    # Limit extremely long text to prevent spaCy memory errors
    if len(text) > 1000000:
        text = text[:1000000]
        
    doc = nlp(text)

    # STEP 1: Contact Info
    matcher = Matcher(nlp.vocab)
    
    # ========== DEFINE PHONE NUMBER PATTERNS ==========
    # These pattern components are reusable building blocks for matching phone number digits
    # in various formats. They follow spaCy's token pattern syntax where each dict represents
    # matching criteria for a single token.
    
    d_3   = {"IS_DIGIT": True, "LENGTH": 3}                   # Matches any token that: is purely digits AND has length of exactly 3
    d_4   = {"IS_DIGIT": True, "LENGTH": 4}                   # Matches any token that: is purely digits AND has length of exactly 4
    d_var = {"IS_DIGIT": True, "LENGTH": {">=": 1, "<=": 4}}  # Matches any token that: is purely digits AND length is between 1-4 digits
    sep   = {"ORTH": {"IN": ["-", ".", "/"]}, "OP": "?"}      # Matches optional separator: hyphen (-), dot (.), or slash (/)
                                                              # "OP": "?" means this pattern is optional (0 or 1 occurrence)

    # Define multiple phone number patterns to catch various international and domestic formats
    # Each sub-list represents a sequence of tokens that make up one phone format pattern
    phone_patterns = [
        # Pattern 1: (123) 456-7890 - U.S. format with parentheses around area code
        # Breakdown: "(" + 3-digits + ")" + optional-sep + 3-digits + optional-sep + 4-digits
        [{"ORTH": "("}, d_3, {"ORTH": ")"}, sep, d_3, sep, d_4],
        
        # Pattern 2: 123-456-7890 or 123.456.7890 - U.S. format without parentheses
        # Breakdown: 3-digits + optional-sep + 3-digits + optional-sep + 4-digits
        [d_3, sep, d_3, sep, d_4],
        
        # Pattern 3: +1-234-567-8901 - International format with country code
        # Breakdown: "+" + 1-4-digits + optional-sep + 1-4-digits + optional-sep + 1-4-digits + optional-sep + 3-9-digits
        [{"ORTH": "+"}, d_var, sep, d_var, sep, d_var, sep, {"IS_DIGIT": True, "LENGTH": {">=": 3, "<=": 9}}],
        
        # Pattern 4: +1(234)567-8901 - International format with country code and parentheses
        # Breakdown: "+" + 1-4-digits + optional-sep + "(" + 3-digits + ")" + optional-sep + 3-digits + optional-sep + 4-digits
        [{"ORTH": "+"}, d_var, sep, {"ORTH": "("}, d_3, {"ORTH": ")"}, sep, d_3, sep, d_4],
        
        # Pattern 5: 10-15 digit continuous phone number (mobile/landline with varying formats worldwide)
        # Some phone numbers are written as continuous digits without separators
        [{"IS_DIGIT": True, "LENGTH": {">=": 10, "<=": 15}}]
    ]
    
    # Register all phone patterns with the matcher under the label "PHONE"
    # The matcher will now search for these patterns in the text
    matcher.add("PHONE", phone_patterns)
    
    # DETECT CONTACT INFO
    # Check for email addresses using spaCy's built-in email detection attribute
    # The like_email attribute recognizes standard email format (word@domain.extension)
    has_email = any(token.like_email for token in doc)
    
    # Check for phone numbers by running the matcher against the document
    # If matcher finds any matches, has_phone will be True
    # len(matcher(doc)) > 0 means: "if there's at least one phone number match, it's a resume signal"
    has_phone = len(matcher(doc)) > 0

    # STEP 2: Headers
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
        [{"LOWER": {"IN": ["references", "languages", "publications", "interests"]}}]
    ]
    header_matcher.add("RESUME_HEADER", header_patterns)
    
    header_matches = header_matcher(doc)
    found_keywords = set()
    for _, start, end in header_matches:
        found_keywords.add(doc[start:end].text.lower())
    keyword_count = len(found_keywords)

    # STEP 3: Entity Density
    entity_count    = 0
    relevant_labels = ["DATE", "ORG", "GPE", "PERSON"]
    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            entity_count += 1

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
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        
        if not text.strip():
            raise ValueError
    except ValueError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF: {pdf_path}, no text could be extracted.",
            "code"   : 503
        }
        print(json.dumps(error_response))
        sys.exit()

    return text

def load_model():
    """
    Loads the spaCy model with error handling
    """
    print("Loading spaCy model ... ")
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
    pdf_path   = os.path.join(script_dir, "Resumes/Resume_own.pdf")

    # STEP 1: Check File Existence
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF not found: {pdf_path}",
            "code"   : 502
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 2: Load Model
    nlp = load_model()

    # STEP 3: Extract Text
    raw_text = extract_text_from_pdf(pdf_path)

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