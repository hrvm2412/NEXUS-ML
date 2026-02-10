import json
import os
import pymupdf
import spacy
import sys

from spacy.matcher import Matcher

class FileExceedsModelLimitError(Exception):
    pass

class FileExceedsPageLimitError(Exception):
    pass

class FileNotResumeError(Exception):
    pass

def clean_and_save_text(text, nlp):
    """
    Removes PII and saves the cleaned text
    """
    print("Preprocessing text (Cleaning & PII Removal) ...")
    
    # Re-use patterns for cleaning (Philippine formats only)
    # Covers all requested formats
    matcher = Matcher(nlp.vocab)

    # Same patterns as detect_resume code, redefined here for scope
    sep = {"ORTH": {"IN": ["-", " "]}, "OP": "?"}
    digit_1 = {"IS_DIGIT": True, "LENGTH": 1}
    digit_2 = {"IS_DIGIT": True, "LENGTH": 2}
    digit_3 = {"IS_DIGIT": True, "LENGTH": 3}
    digit_4 = {"IS_DIGIT": True, "LENGTH": 4}
    digit_10 = {"IS_DIGIT": True, "LENGTH": 10}
    digit_11 = {"IS_DIGIT": True, "LENGTH": 11}
    
    phone_patterns = [
        [digit_11],
        [digit_4, sep, digit_3, sep, digit_4],
        [{"ORTH": "+63"}, digit_10],
        [{"ORTH": "+63"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_10],
        [{"IS_DIGIT": True, "LENGTH": 2}, digit_10],
        [{"IS_DIGIT": True, "LENGTH": 2}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        [{"ORTH": "("}, {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": ")"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        [{"ORTH": "("}, {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": ")"}, sep, digit_10]
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
    
    # Tokenize and lowercase for final output
    print("Tokenizing and lowercasing ...")
    doc = nlp(cleaned_text_with_structure)
    cleaned_tokens = [token.text.lower() for token in doc if not token.is_space]
    cleaned_text = " ".join(cleaned_tokens)

    # Apply smart casing preprocessing for extraction
    preprocessed_text = preprocess_for_extraction(cleaned_text_with_structure, nlp)

    # Save both versions
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Tokenized and lowercased (restored behavior)
    output_path_cleaned = os.path.join(output_dir, "Resume_cleaned.txt")
    with open(output_path_cleaned, "w", encoding = "utf-8") as f:
        f.write(cleaned_text)

    # 2. Preprocessed for extraction (smart casing applied)
    output_path_for_extraction = os.path.join(output_dir, "Resume_for_extraction.txt")
    with open(output_path_for_extraction, "w", encoding = "utf-8") as f:
        f.write(preprocessed_text)
        
    print(f"Cleaned text (tokenized/lowercased) saved to: {output_path_cleaned}")
    print(f"Preprocessed text (smart casing) saved to: {output_path_for_extraction}")

def detect_resume(text, nlp):
    """
    Analyzes raw text to determine if it is a resume
    """
    print("Analyzing text for resume detection ...")

    try:
        if len(text) > 1000000:
            raise FileExceedsModelLimitError
    except FileExceedsModelLimitError:
        error_response = {
            "status" : "error",
            "message": "Resume text exceeds spaCy model limit (1MB maximum).",
            "code"   : 400
        }
        print(json.dumps(error_response))
        sys.exit()
        
    doc = nlp(text)

    # STEP 1: Contact Info
    matcher = Matcher(nlp.vocab)
    
                                                        # Philippine mobile phone number patterns ONLY
                                                        # Reusable components for flexible matching
    sep      = {"ORTH": {"IN": ["-", " "]}, "OP": "?"}  # Optional separator: hyphen or space
    digit_1  = {"IS_DIGIT": True, "LENGTH": 1}          # Single digit (e.g., 9)
    digit_2  = {"IS_DIGIT": True, "LENGTH": 2}          # 2 digits (e.g., XX)
    digit_3  = {"IS_DIGIT": True, "LENGTH": 3}          # 3 digits (e.g., XXX)
    digit_4  = {"IS_DIGIT": True, "LENGTH": 4}          # 4 digits (e.g., XXXX or 09XX)
    digit_10 = {"IS_DIGIT": True, "LENGTH": 10}         # 10 digits (9XXXXXXXXX)
    digit_11 = {"IS_DIGIT": True, "LENGTH": 11}         # 11 digits (09XXXXXXXXX)
    
    # Define Philippine phone number patterns
    phone_patterns = [
        # Pattern 1: 09XXXXXXXXX - Local continuous (11 digits total)
        [digit_11],
        
        # Pattern 2: 09XX-XXX-XXXX or 09XX XXX XXXX - Local with flexible separators
        [digit_4, sep, digit_3, sep, digit_4],

        # Pattern 3: +639XXXXXXXXX - Continuous (no spaces/dashes)
        [{"ORTH": "+63"}, digit_10],
        
        # Pattern 4: +63 9XX XXX XXXX or +63-9XX-XXX-XXXX - With flexible separators
        [{"ORTH": "+63"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Pattern 5: (+63) 9XX XXX XXXX - Parentheses + flexible separators
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Pattern 6: (+63) 9XXXXXXXXX or (+63)-9XXXXXXXXX - Parentheses + continuous/dash
        [{"ORTH": "("}, {"ORTH": "+63"}, {"ORTH": ")"}, sep, digit_10],

        # Pattern 7: 639XXXXXXXXX - Continuous (11 digits total: 63 + 9 + 8)
        [{"IS_DIGIT": True, "LENGTH": 2}, digit_10],
        
        # Pattern 8: 63 9XX XXX XXXX or 63-9XX-XXX-XXXX - With flexible separators
        [{"IS_DIGIT": True, "LENGTH": 2}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Pattern 9: (63) 9XX XXX XXXX - Parentheses + flexible separators
        [{"ORTH": "("}, {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": ")"}, sep, digit_1, digit_2, sep, digit_3, sep, digit_4],
        
        # Pattern 10: (63) 9XXXXXXXXX or (63)-9XXXXXXXXX - Parentheses + continuous/dash
        [{"ORTH": "("}, {"IS_DIGIT": True, "LENGTH": 2}, {"ORTH": ")"}, sep, digit_10]
    ]
    
    # Register all phone patterns with the matcher under the label "PHONE"
    # The matcher will now search for these patterns in the text
    matcher.add("PHONE", phone_patterns)
    
    # Step 2: Email Detection
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
        [{"LOWER": {"IN": ["references", "languages", "publications", "interests"]}}],
        [{"LOWER": {"IN": ["volunteering", "volunteer", "volunteer experience", "community service"]}}],
        [{"LOWER": {"IN": ["memberships", "professional memberships", "associations", "affiliations"]}}],
        [{"LOWER": {"IN": ["training", "workshops", "professional development", "courses"]}}],
        [{"LOWER": {"IN": ["patents", "publications", "research", "intellectual property"]}}],
        [{"LOWER": {"IN": ["honors", "distinctions", "recognitions", "scholarships"]}}],
        [{"LOWER": {"IN": ["internship", "internships", "practicum"]}}],
        [{"LOWER": {"IN": ["core competencies", "key competencies", "core strengths"]}}],
        [{"LOWER": {"IN": ["strengths", "personal strengths", "key strengths"]}}]
    ]
    header_matcher.add("RESUME_HEADER", header_patterns)
    
    # Search the text for all resume section headers
    header_matches = header_matcher(doc)
    
    # Create a list to store the header keywords we find (without repeats)
    found_keywords = set()
    
    # Go through each match and save the header text
    for _, start, end in header_matches:
        found_keywords.add(doc[start:end].text.lower())
    
    # Count how many different headers we found
    keyword_count = len(found_keywords)

    # STEP 3: Entity Density
    # Start with zero important items found
    entity_count    = 0
    # List of important item types to look for (dates, companies, locations, people)
    relevant_labels = ["DATE", "ORG", "GPE", "PERSON"]
    # Go through all important items detected in the text
    for ent in doc.ents:
        # If this item is one of the types we care about, add it to our count
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
            "code"   : 300
        }
        print(json.dumps(error_response))
        sys.exit()
    except ValueError:
        error_response = {
            "status" : "error",
            "message": f"Resume PDF: {pdf_path}, no text could be extracted.",
            "code"   : 301
        }
        print(json.dumps(error_response))
        sys.exit()
    print("Text extracted from PDF.")

    return text

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
            "code"   : 200
        }
        print(json.dumps(error_response))
        sys.exit()
    else:
        print("Model loaded.")
        return nlp

def preprocess_for_extraction(text, nlp):
    """
    Preprocess text for skill/knowledge extraction using context-aware casing:
    - Preserves Acronyms (e.g., SQL, AWS)
    - Preserves Proper Nouns (e.g., Python, Docker, Google)
    - Lowercases common words (verbs, adjectives), even at the start of sentences
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
            if token.text.isupper() and len(token.text) > 1:
                processed += token.text + token.whitespace_
            
            # RULE 2: Proper Nouns (Keep Title Case)
            elif token.pos_ == "PROPN" or token.ent_type_ in preserve_ent_types:
                processed += token.text + token.whitespace_
                
            # RULE 3: Common Words (Lowercase)
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
    nlp = load_model()

    # STEP 3: Extract Text
    raw_text = extract_text_from_pdf(pdf_path)

    # STEP 4: Detect Resume
    try:
        if detect_resume(raw_text, nlp):
            print("This file is a resume.")
    except FileNotResumeError:
        error_response = {
            "status" : "error",
            "message": "This file is NOT a resume.",
            "code"   : 101
        }
        print(json.dumps(error_response))
        sys.exit()

    # STEP 5: Clean and Save
    clean_and_save_text(raw_text, nlp)