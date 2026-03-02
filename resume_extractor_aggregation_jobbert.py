import gc
import json
import os
import re
import sys
import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Define local model paths
SKILL_MODEL_PATH     = "./Models/jobbert_skill_extraction"
KNOWLEDGE_MODEL_PATH = "./Models/jobbert_knowledge_extraction"

# HuggingFace model IDs (for downloading if needed)
SKILL_MODEL_ID     = "jjzha/jobbert_skill_extraction"
KNOWLEDGE_MODEL_ID = "jjzha/jobbert_knowledge_extraction"

# Confidence threshold for filtering predictions
CONFIDENCE_THRESHOLD = 0.7

# Sliding window configuration
WINDOW_SIZE = 2  # Number of lines to process together

def capitalize_words(text):
    """
    Capitalize every word in the text for display
    Words that are already capitalized (have any uppercase letters) are preserved as-is
    """
    words = text.split()
    capitalized_words = []
    
    for word in words:
        # If word already has any uppercase letters, preserve it as-is
        if any(c.isupper() for c in word):
            capitalized_words.append(word)
        else:
            # Word is all lowercase, capitalize it
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

def check_token_limit(text, tokenizer, max_tokens):
    """Check if text exceeds token limit"""
    tokens = tokenizer.encode(text, add_special_tokens = True)
    return len(tokens) <= max_tokens, len(tokens)

def clean_entity_text(text):
    """Remove parentheses/brackets from any position, and leading/trailing punctuation"""
    # Remove parentheses and brackets from anywhere in the text
    cleaned = re.sub(r'[(){}\[\]]', '', text).strip()
    
    # Remove trailing punctuation only (no parentheses/brackets)
    while cleaned and cleaned[-1] in ',.;:!?':
        cleaned = cleaned[:-1].strip()
    
    # Remove leading punctuation only (no parentheses/brackets)
    while cleaned and cleaned[0] in ',.;:!?':
        cleaned = cleaned[1:].strip()
    
    return cleaned

# Remove duplicates while preserving order (keep highest score) 
def deduplicate_entities(entities):
    """Remove duplicate entities, keeping the one with highest score"""
    seen = {}
    for entity in entities:
        text_lower = entity['text'].lower()
        
        # Check if we've seen this skill before (case-insensitive)
        # Example: First we see "Java" (score 0.85) from Summary. Later "Java" (score 0.95) from Experience
        if text_lower not in seen or entity['score'] > seen[text_lower]['score']:
            # If it's new, add it. If it exists but this one is better, replace it
            # Result: We keep the instance with 0.95 confidence
            seen[text_lower] = entity
    return list(seen.values())

def determine_primary_line(entity_start, entity_end, line_starts, window_lines):
    """
    Determine which line number an entity primarily belongs to based on character position
    """
    # Find which line the majority of the entity belongs to
    line_contributions = []
    
    for idx, (line_num, line_start) in enumerate(zip(window_lines, line_starts)):
        # Determine where this line ends (it ends where the next line begins)
        if idx < len(line_starts) - 1:
            line_end = line_starts[idx + 1]
        else:
            # The last line goes until the end of the text
            line_end = float('inf')
        
        # Calculate overlap between entity and this line
        # We want to know: How many characters of this entity are actually on this line
        overlap_start = max(entity_start, line_start)
        overlap_end   = min(entity_end, line_end)
        overlap       = max(0, overlap_end - overlap_start)
        
        line_contributions.append((line_num, overlap))
    
    # Return the line with maximum contribution
    # Example: If "Machine Learning" has 7 chars on line 1 and 9 chars on line 2, assign to line 2
    primary_line = max(line_contributions, key = lambda x: x[1])[0]
    return primary_line

def download_model_if_needed(model_path, model_id):
    """Download model from HuggingFace if not exists locally"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Downloading {model_id} from HuggingFace ...")
        try:
            # Create directory
            os.makedirs(model_path, exist_ok = True)

            # Download model and tokenizer
            model     = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Save locally
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"✓ Model downloaded and saved to {model_path}")

            # Free download objects — local files are the source of truth from here
            del model, tokenizer
            gc.collect()
        except Exception as e:
            error_response = {
                "status" : "error",
                "message": f"Failed to download model {model_id}: {str(e)}",
                "code"   : 300
            }
            print(json.dumps(error_response))
            sys.exit()
    else:
        print(f"Model found at {model_path}")

def expand_to_word_boundaries(text, start, end):
    """Expand the entity boundaries to include complete words."""
    # Expand left to include beginning of word
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in ',.;:!?()[]{}/-':
        start -= 1
    
    # Expand right to include end of word
    while end < len(text) and not text[end].isspace() and text[end] not in ',.;:!?()[]{}/-':
        end += 1
    
    return start, end

def filter_by_confidence(entities, threshold = CONFIDENCE_THRESHOLD):
    """Filter entities by confidence score threshold"""
    return [e for e in entities if e['score'] >= threshold]

def is_valid_entity(text):
    """Check if entity is valid (not just punctuation or parentheses)"""
    # 1. Minimum length check
    # Example: "A" (too short), "" (empty) then rejected
    if not text or len(text) < 2:
        return False
    
    # 2. Content check
    # Must contain at least one alphanumeric character (letter or number)
    # Example: "..." or "---" then rejected (only punctuation). "C++" then accepted (contains 'C')
    if not any(c.isalnum() for c in text):
        return False
    
    # 3. Specific symbol check
    # Should not be only parentheses or brackets
    # Example: "()" or "[ ]" then Rejected (sometimes NER picks up empty brackets)
    if text.strip() in ['(', ')', '()', '[]', '{}', '( )', '[ ]', '{ }']:
        return False
    
    return True

def merge_adjacent_entities(entities, original_text, max_gap = 2):
    """Merge entities that are directly adjacent (for fixing word boundary issues)"""
    if not entities:
        return []
    
    merged = []
    i      = 0
    
    # Iterate through the list of entities
    while i < len(entities):
        current = entities[i]
        
        # Look ahead to see if the next entity is close enough to merge
        while i + 1 < len(entities):
            next_entity = entities[i + 1]
            # Calculate distance between end of current entity and start of next
            gap         = next_entity['start'] - current['end']
            
            # Only merge if gap is very small (1-2 characters)
            if gap <= max_gap:
                between_text = original_text[current['end']:next_entity['start']]
                
                # CRITICAL: Only merge if between text is ONLY whitespace
                # Example: "Machine Learning" (Merge) vs "Java, Python" (Don't Merge)
                if between_text.strip() == '':
                    # Additional check: make sure there's no punctuation nearby
                    has_separator = any(sep in between_text for sep in ',;:|')
                    
                    if not has_separator:
                        # Merge the next entity into the current one
                        current['text']   = original_text[current['start']:next_entity['end']]
                        current['end']    = next_entity['end']
                        # Average the scores of the two parts
                        current['score']  = (current['score'] + next_entity['score']) / 2
                        # Skip the next entity since it's now merged
                        i                += 1
                    else:
                        break
                else:
                    # If there's any non-whitespace between entities, don't merge
                    break
            else:
                break
        
        merged.append(current)
        i += 1
    
    return merged

def merge_bio_entities(results, original_text):
    """Merge B and I tags into complete entities with word boundary expansion"""
    if not results:
        return []
    
    merged         = []
    current_entity = None
    
    # Loop through every token the model found
    for entity in results:
        label = entity['entity_group']
        
        # 'B' means Beginning of a new entity
        if label == 'B':
            # If we were building an entity, save it now because a new one is starting
            if current_entity:
                merged.append(current_entity)
            
            # Expand to word boundaries to fix partial words
            # Example: Model detects "Py" then expands to "Python"
            start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
            
            # Start tracking this new entity
            current_entity = {
                'text' : original_text[start:end],
                'start': start,
                'end'  : end,
                'score': entity['score']
            }
        # 'I' means Inside (continuation) of the current entity
        elif label == 'I' and current_entity:
            # Check if there's a separator between current entity end and this token
            between_text = original_text[current_entity['end']:entity['start']]
            
            # Stop merging if there's a comma, semicolon, or pipe separator
            # This handles cases where the model missed the split (e.g., "Java, Python")
            # Example: Current = "Java", Next = "Python", Separator="," in which: Save "Java", Start "Python"
            if any(sep in between_text for sep in [',', ';', '|', '&']):
                # Save current entity and start a new one
                merged.append(current_entity)
                
                # Treat this I-tag as a new entity (edge case handling)
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                current_entity = {
                    'text' : original_text[start:end],
                    'start': start,
                    'end'  : end,
                    'score': entity['score']
                }
            else:
                # Ensure we capture the full word of this new segment before merging
                # Example: Current = "Machine", Next = "Learning" then merge to "Machine Learning"
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                
                # Update the existing entity to include this new segment (merging them)
                current_entity['text']  = original_text[current_entity['start']:end]
                current_entity['end']   = end
                current_entity['score'] = (current_entity['score'] + entity['score']) / 2
    
    if current_entity:
        merged.append(current_entity)
    
    return merged

def remove_substring_duplicates(entities):
    """
    Remove entities that are substrings of other entities on the same line
    Keeps the longer/more complete entity
    
    Examples:
    - "Digital Video" vs "Digital Video Switcher" (same line) to keep "Digital Video Switcher"
    - "Adobe Premiere" vs "Adobe Premiere Pro" (same line) to keep "Adobe Premiere Pro"
    - "Machine" vs "Machine Learning" (same line) to keep "Machine Learning"
    """
    if not entities:
        return []
    
    # Group entities by line number
    by_line = {}
    for entity in entities:
        line = entity['line']
        if line not in by_line:
            by_line[line] = []
        by_line[line].append(entity)
    
    filtered = []
    
    for line_num, line_entities in by_line.items():
        # Sort by length (descending) so we check longer strings first
        line_entities.sort(key = lambda x: len(x['text']), reverse = True)
        
        keep = []
        for i, entity in enumerate(line_entities):
            is_substring = False
            
            # Check if this entity is a substring of any other entity in keep list
            for kept_entity in keep:
                if entity['text'].lower() in kept_entity['text'].lower():
                    is_substring = True
                    break
            
            if not is_substring:
                keep.append(entity)
        
        filtered.extend(keep)
    
    return filtered

def split_entities_by_separators(entities, original_text):
    """Post-processing: Split any entities that still contain commas or list separators"""
    split_entities = []
    
    for entity in entities:
        text = entity['text']
        
        # Check if entity contains list separators
        # Example: Model extracted "Python, Java" as a single skill instead of two
        if any(sep in text for sep in [',', ';', '|']):
            # Split by common separators
            # Example: "Python, Java" -> ["Python", "Java"]
            parts = re.split(r'[,;|]', text)
            
            # Track position in original text so we don't match the wrong occurrence
            current_pos = entity['start']
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # Find this part in the original text
                # We need to find where "Python" and "Java" actually sit in the text to get correct start/end indices
                part_start = original_text.find(part, current_pos)
                if part_start != -1:
                    part_end = part_start + len(part)
                    
                    split_entities.append({
                        'text' : part,
                        'start': part_start,
                        'end'  : part_end,
                        'score': entity['score']
                    })
                    
                    # Move past this part so next search starts after it
                    current_pos = part_end
        else:
            # No separators, keep as is
            split_entities.append(entity)
    
    return split_entities

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Download models if needed
    print("Checking models ...")

    download_model_if_needed(SKILL_MODEL_PATH, SKILL_MODEL_ID)
    download_model_if_needed(KNOWLEDGE_MODEL_PATH, KNOWLEDGE_MODEL_ID)

    # 2. Initialize pipelines and tokenizer
    print("Initializing pipelines ...")

    try:
        # Initialize pipelines for NER (Named Entity Recognition)
        skill_extractor = pipeline(
            "token-classification",
            model                = SKILL_MODEL_PATH,
            tokenizer            = SKILL_MODEL_PATH,
            aggregation_strategy = "simple",
            device               = device
        )

        knowledge_extractor = pipeline(
            "token-classification",
            model                = KNOWLEDGE_MODEL_PATH,
            tokenizer            = KNOWLEDGE_MODEL_PATH,
            aggregation_strategy = "simple",
            device               = device
        )
        
        # Load tokenizers separately for token counting
        skill_tokenizer     = AutoTokenizer.from_pretrained(SKILL_MODEL_PATH)
        knowledge_tokenizer = AutoTokenizer.from_pretrained(KNOWLEDGE_MODEL_PATH)
        
        print("Models loaded successfully from local storage.")
    except Exception as e:
        error_response = {
            "status" : "error",
            "message": f"Failed to load models: {str(e)}",
            "code"   : 301
        }
        print(json.dumps(error_response))
        sys.exit()

    # Get the maximum number of tokens (usually 512 for BERT)
    # We take the minimum of both models to ensure safety, though they are likely identical
    MAX_TOKENS = min(skill_tokenizer.model_max_length, knowledge_tokenizer.model_max_length)

    # Tokenizers only needed for MAX_TOKENS — free them before the extraction loop
    del skill_tokenizer, knowledge_tokenizer
    gc.collect()

    # Print configuration details for debugging/logging
    print(f"Model max tokens: {MAX_TOKENS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Sliding window size: {WINDOW_SIZE} lines")
    print("Pipelines initialized successfully.")

    # 3. Load preprocessed resume text file
    txt_path = "Resume_text/Resume_for_extraction.txt"

    try:
        if not os.path.exists(txt_path):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Preprocessed resume text file not found: {txt_path}. Run resume_text_detector_cleaner.py first.",
            "code"   : 302
        }
        print(json.dumps(error_response))
        sys.exit()

    with open(txt_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()

    # Remove empty lines and strip
    lines = [line.strip() for line in lines if line.strip()]

    # Store all extracted skills and knowledge
    all_skills         = []
    all_knowledge      = []
    processed_entities = set()  # To avoid duplicates from overlapping windows

    # 4. Process lines using sliding window
    print(f"Processing {len(lines)} lines with sliding window (size = {WINDOW_SIZE}) ...")

    for window_start in range(0, len(lines), 1):  # Slide by 1 line each time
        # Get window of lines
        window_end           = min(window_start + WINDOW_SIZE, len(lines))
        window_lines_indices = list(range(window_start, window_end))
        
        if len(window_lines_indices) < 1:
            break
        
        # Get the actual lines
        window_lines_text = [lines[i] for i in window_lines_indices]
        
        # Combine lines with newline separator for better model context
        combined_text = '\n'.join(window_lines_text)
        
        # We need to track where each line begins (character index) within the combined text block
        # This helps us map a found skill back to its specific line number later
        line_starts = [0]
        current_pos = 0
        for line_text in window_lines_text[:-1]:
            current_pos += len(line_text) + 1  # Add length of line + 1 for the newline character (\n)
            line_starts.append(current_pos)
        
        # Convert computer counting (0, 1, 2) to human counting (1, 2, 3) for the final report
        window_line_nums = [i + 1 for i in range(window_start, window_end)]
        del window_lines_text, current_pos

        # Check token limits
        skill_within_limit    , skill_token_count     = check_token_limit(combined_text, skill_extractor.tokenizer, MAX_TOKENS)
        knowledge_within_limit, knowledge_token_count = check_token_limit(combined_text, knowledge_extractor.tokenizer, MAX_TOKENS)
        
        if not skill_within_limit or not knowledge_within_limit:
            error_response = {
                "status" : "error",
                "message": f"Text window exceeds model token limit ({MAX_TOKENS}).",
                "code"   : 303
            }
            print(json.dumps(error_response))
            sys.exit()
        
        # 5. Extract Skills & Knowledge
        # 5.1 Run the NER model on the text to find potential skills
        skill_results = skill_extractor(combined_text)
        # 5.2 Combine 'Begin' and 'Inside' tags (BIO scheme) into full words/phrases
        skill_results = merge_bio_entities(skill_results, combined_text)
        # 5.3 Connect adjacent entities that should be one (e.g., "Machine" + "Learning")
        skill_results = merge_adjacent_entities(skill_results, combined_text)
        # 5.4 Split entities that accidentally include commas (e.g., "Python, Java")
        skill_results = split_entities_by_separators(skill_results, combined_text)
        # 5.5 Remove weak predictions below our confidence threshold
        skill_results = filter_by_confidence(skill_results, CONFIDENCE_THRESHOLD)
        
        for entity in skill_results:
            # Clean up punctuation (e.g., remove trailing commas)
            cleaned_text = clean_entity_text(entity['text'])
            
            # Validate: Ensure it's not just symbols or single characters
            if cleaned_text and is_valid_entity(cleaned_text):
                # Format: Capitalize words for consistent display (e.g. "machine learning" to "Machine Learning")
                capitalized_text = capitalize_words(cleaned_text)
                # Map the entity back to the specific line number in the original text
                primary_line     = determine_primary_line(entity['start'], entity['end'], line_starts, window_line_nums)
                
                # Create a unique key to prevent duplicate entries for the same skill on the same line
                # This is necessary because the sliding window overlaps, causing the same line to be processed multiple times
                entity_key       = (capitalized_text.lower(), primary_line)
                
                if entity_key not in processed_entities:
                    processed_entities.add(entity_key)
                    all_skills.append({
                        'text' : capitalized_text,
                        'score': float(entity['score']),
                        'line' : primary_line
                    })

        del skill_results
        gc.collect()
        
        # Extract Knowledge (same pipeline as skills)
        knowledge_results = knowledge_extractor(combined_text)
        knowledge_results = merge_bio_entities(knowledge_results, combined_text)
        knowledge_results = merge_adjacent_entities(knowledge_results, combined_text)
        knowledge_results = split_entities_by_separators(knowledge_results, combined_text)
        knowledge_results = filter_by_confidence(knowledge_results, CONFIDENCE_THRESHOLD)
        
        for entity in knowledge_results:
            cleaned_text = clean_entity_text(entity['text'])
            if cleaned_text and is_valid_entity(cleaned_text):
                capitalized_text = capitalize_words(cleaned_text)
                primary_line     = determine_primary_line(entity['start'], entity['end'], line_starts, window_line_nums)
                entity_key       = (capitalized_text.lower(), primary_line)
                
                if entity_key not in processed_entities:
                    processed_entities.add(entity_key)
                    all_knowledge.append({
                        'text' : capitalized_text,
                        'score': float(entity['score']),
                        'line' : primary_line
                    })

        del knowledge_results, combined_text, line_starts, window_line_nums
        gc.collect()

    # Pipelines and source lines no longer needed after the extraction loop
    del skill_extractor, knowledge_extractor, lines, processed_entities
    gc.collect()

    # 6. Global Deduplication
    # First, remove exact duplicates (same text) across the whole file
    all_skills    = deduplicate_entities(all_skills)
    all_knowledge = deduplicate_entities(all_knowledge)

    # 7. Substring Cleanup
    # Remove shorter skills that are already part of longer ones on the same line
    # Example: If we have "Excel" and "Microsoft Excel" on line 5, remove "Excel"
    print("Removing substring duplicates ...")
    all_skills    = remove_substring_duplicates(all_skills)
    all_knowledge = remove_substring_duplicates(all_knowledge)

    # Sort by score (descending)
    all_skills.sort(key = lambda x: x['score'], reverse = True)
    all_knowledge.sort(key = lambda x: x['score'], reverse = True)

    # 8. Save skills and knowledge to JSON — parentheses stripped from text
    output = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "window_size"         : WINDOW_SIZE,
        "skills": [
            {
                "text" : skill['text'],
                "score": round(skill['score'], 4),
                "line" : skill['line']
            }
            for skill in all_skills
        ],
        "knowledge": [
            {
                "text" : knowledge['text'],
                "score": round(knowledge['score'], 4),
                "line" : knowledge['line']
            }
            for knowledge in all_knowledge
        ]
    }

    output_path = "Extracted_skills_knowledge/Resume_skills_knowledge.json"
    try:
        with open(output_path, 'w', encoding = 'utf-8') as f:
            json.dump(output, f, indent = 4, ensure_ascii = False)
    except Exception as e:
        error_response = {
            "status" : "error",
            "message": f"Failed to export JSON file: {str(e)}",
            "code"   : 304
        }
        print(json.dumps(error_response))
        sys.exit()

    print("Extraction complete!")