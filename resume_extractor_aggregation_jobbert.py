# AI
import torch
import os
import sys
import json
import re

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Define local model paths
SKILL_MODEL_PATH     = "./models/jobbert_skill_extraction"
KNOWLEDGE_MODEL_PATH = "./models/jobbert_knowledge_extraction"

# HuggingFace model IDs (for downloading if needed)
SKILL_MODEL_ID     = "jjzha/jobbert_skill_extraction"
KNOWLEDGE_MODEL_ID = "jjzha/jobbert_knowledge_extraction"

# Confidence threshold for filtering predictions
CONFIDENCE_THRESHOLD = 0.7

# Sliding window configuration
WINDOW_SIZE = 2  # Number of lines to process together

def download_model_if_needed(model_path, model_id):
    """Download model from HuggingFace if not exists locally"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Downloading {model_id} from HuggingFace...")
        try:
            os.makedirs(model_path, exist_ok=True)
            model = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"✓ Model downloaded and saved to {model_path}")
        except Exception as e:
            error_response = {
                "status": "error",
                "message": f"Failed to download model {model_id}: {str(e)}",
                "code": 503
            }
            print(json.dumps(error_response))
            sys.exit()
    else:
        print(f"✓ Model found at {model_path}")

def check_token_limit(text, tokenizer, max_tokens):
    """Check if text exceeds token limit"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens) <= max_tokens, len(tokens)

def expand_to_word_boundaries(text, start, end):
    """Expand the entity boundaries to include complete words."""
    # Expand left to include beginning of word
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in ',.;:!?()[]{}/-':
        start -= 1
    
    # Expand right to include end of word
    while end < len(text) and not text[end].isspace() and text[end] not in ',.;:!?()[]{}/-':
        end += 1
    
    return start, end

def merge_bio_entities(results, original_text):
    """Merge B and I tags into complete entities with word boundary expansion."""
    if not results:
        return []
    
    merged = []
    current_entity = None
    
    for entity in results:
        label = entity['entity_group']
        
        if label == 'B':
            if current_entity:
                merged.append(current_entity)
            
            # Expand to word boundaries to fix partial words
            start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
            
            current_entity = {
                'text': original_text[start:end],
                'start': start,
                'end': end,
                'score': entity['score']
            }
        elif label == 'I' and current_entity:
            # Check if there's a separator between current entity end and this token
            between_text = original_text[current_entity['end']:entity['start']]
            
            # Stop merging if there's a comma, semicolon, or pipe separator
            if any(sep in between_text for sep in [',', ';', '|', '&']):
                # Save current entity and start a new one
                merged.append(current_entity)
                
                # Treat this I-tag as a new entity (edge case handling)
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                current_entity = {
                    'text': original_text[start:end],
                    'start': start,
                    'end': end,
                    'score': entity['score']
                }
            else:
                # Expand to word boundaries before merging
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                
                # Extend current entity to this position
                current_entity['text'] = original_text[current_entity['start']:end]
                current_entity['end'] = end
                current_entity['score'] = (current_entity['score'] + entity['score']) / 2
    
    if current_entity:
        merged.append(current_entity)
    
    return merged

def merge_adjacent_entities(entities, original_text, max_gap=2):
    """Merge entities that are directly adjacent (for fixing word boundary issues)."""
    if not entities:
        return []
    
    merged = []
    i = 0
    
    while i < len(entities):
        current = entities[i]
        
        while i + 1 < len(entities):
            next_entity = entities[i + 1]
            gap = next_entity['start'] - current['end']
            
            # Only merge if gap is very small (1-2 characters)
            if gap <= max_gap:
                between_text = original_text[current['end']:next_entity['start']]
                
                # CRITICAL: Only merge if between text is ONLY whitespace
                if between_text.strip() == '':
                    # Additional check: make sure there's no punctuation nearby
                    has_separator = any(sep in between_text for sep in ',;:|')
                    
                    if not has_separator:
                        current['text'] = original_text[current['start']:next_entity['end']]
                        current['end'] = next_entity['end']
                        current['score'] = (current['score'] + next_entity['score']) / 2
                        i += 1
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

def filter_by_confidence(entities, threshold=CONFIDENCE_THRESHOLD):
    """Filter entities by confidence score threshold"""
    return [e for e in entities if e['score'] >= threshold]

def split_entities_by_separators(entities, original_text):
    """Post-processing: Split any entities that still contain commas or list separators."""
    split_entities = []
    
    for entity in entities:
        text = entity['text']
        
        # Check if entity contains list separators
        if any(sep in text for sep in [',', ';', '|']):
            # Split by common separators
            parts = re.split(r'[,;|]', text)
            
            # Track position in original text
            current_pos = entity['start']
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # Find this part in the original text
                part_start = original_text.find(part, current_pos)
                if part_start != -1:
                    part_end = part_start + len(part)
                    
                    split_entities.append({
                        'text': part,
                        'start': part_start,
                        'end': part_end,
                        'score': entity['score']
                    })
                    
                    current_pos = part_end
        else:
            # No separators, keep as is
            split_entities.append(entity)
    
    return split_entities

def clean_entity_text(text):
    """Remove leading/trailing punctuation and parentheses."""
    cleaned = text.strip()
    
    # Remove trailing punctuation and parentheses
    while cleaned and cleaned[-1] in ',.;:!?()[]{}':
        cleaned = cleaned[:-1].strip()
    
    # Remove leading punctuation and parentheses
    while cleaned and cleaned[0] in ',.;:!?()[]{}':
        cleaned = cleaned[1:].strip()
    
    # Filter out if only parentheses or empty
    if not cleaned or cleaned in ['(', ')', '()', '[]', '{}']:
        return ''
    
    return cleaned

def is_valid_entity(text):
    """Check if entity is valid (not just punctuation or parentheses)."""
    if not text or len(text) < 2:
        return False
    
    # Must contain at least one alphanumeric character
    if not any(c.isalnum() for c in text):
        return False
    
    # Should not be only parentheses or brackets
    if text.strip() in ['(', ')', '()', '[]', '{}', '( )', '[ ]', '{ }']:
        return False
    
    return True

def capitalize_words(text):
    """
    Capitalize every word in the text for display.
    Words that are already capitalized (have any uppercase letters) are preserved as-is.
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

def determine_primary_line(entity_start, entity_end, line_starts, window_lines):
    """
    Determine which line number an entity primarily belongs to based on character position.
    
    Args:
        entity_start: Start position of entity in combined text
        entity_end: End position of entity in combined text
        line_starts: List of starting positions for each line in combined text
        window_lines: List of original line numbers in this window
    
    Returns:
        Primary line number (1-indexed)
    """
    # Find which line the majority of the entity belongs to
    entity_length = entity_end - entity_start
    line_contributions = []
    
    for idx, (line_num, line_start) in enumerate(zip(window_lines, line_starts)):
        if idx < len(line_starts) - 1:
            line_end = line_starts[idx + 1]
        else:
            line_end = float('inf')
        
        # Calculate overlap between entity and this line
        overlap_start = max(entity_start, line_start)
        overlap_end = min(entity_end, line_end)
        overlap = max(0, overlap_end - overlap_start)
        
        line_contributions.append((line_num, overlap))
    
    # Return the line with maximum contribution
    primary_line = max(line_contributions, key=lambda x: x[1])[0]
    return primary_line

# Remove duplicates while preserving order (keep highest score)
def deduplicate_entities(entities):
    """Remove duplicate entities, keeping the one with highest score"""
    seen = {}
    for entity in entities:
        text_lower = entity['text'].lower()
        if text_lower not in seen or entity['score'] > seen[text_lower]['score']:
            seen[text_lower] = entity
    return list(seen.values())

def remove_substring_duplicates(entities):
    """
    Remove entities that are substrings of other entities on the same line.
    Keeps the longer/more complete entity.
    
    Examples:
    - "Digital Video" vs "Digital Video Switcher" (same line) → Keep "Digital Video Switcher"
    - "Adobe Premiere" vs "Adobe Premiere Pro" (same line) → Keep "Adobe Premiere Pro"
    - "Machine" vs "Machine Learning" (same line) → Keep "Machine Learning"
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
        line_entities.sort(key=lambda x: len(x['text']), reverse=True)
        
        keep = []
        for i, entity in enumerate(line_entities):
            is_substring = False
            
            # Check if this entity is a substring of any other entity in keep list
            for kept_entity in keep:
                if entity['text'].lower() in kept_entity['text'].lower():
                    # This entity is a substring of a kept entity, skip it
                    is_substring = True
                    print(f"  [DEDUP] Removing '{entity['text']}' (substring of '{kept_entity['text']}' on Line {line_num})")
                    break
            
            if not is_substring:
                keep.append(entity)
        
        filtered.extend(keep)
    
    return filtered

# 1. Initialize pipelines and tokenizer
print("Initializing pipelines...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download models if needed
print("Checking models...")
download_model_if_needed(SKILL_MODEL_PATH, SKILL_MODEL_ID)
download_model_if_needed(KNOWLEDGE_MODEL_PATH, KNOWLEDGE_MODEL_ID)

# Load models from local paths
try:
    skill_extractor = pipeline(
        "token-classification",
        model=SKILL_MODEL_PATH,
        tokenizer=SKILL_MODEL_PATH,
        aggregation_strategy="simple",
        device=device
    )
    knowledge_extractor = pipeline(
        "token-classification",
        model=KNOWLEDGE_MODEL_PATH,
        tokenizer=KNOWLEDGE_MODEL_PATH,
        aggregation_strategy="simple",
        device=device
    )
    
    skill_tokenizer = AutoTokenizer.from_pretrained(SKILL_MODEL_PATH)
    knowledge_tokenizer = AutoTokenizer.from_pretrained(KNOWLEDGE_MODEL_PATH)
    
    print("✓ Models loaded successfully from local storage")
except Exception as e:
    error_response = {
        "status": "error",
        "message": f"Failed to load models: {str(e)}",
        "code": 504
    }
    print(json.dumps(error_response))
    sys.exit()

MAX_TOKENS = skill_tokenizer.model_max_length
print(f"Model max tokens: {MAX_TOKENS}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Sliding window size: {WINDOW_SIZE} lines")
print("Pipelines initialized successfully.\n")

# 2. Load preprocessed resume text file
txt_path = "Resume_preprocessed_no_space.txt"

try:
    if not os.path.exists(txt_path):
        raise FileNotFoundError
except FileNotFoundError:
    error_response = {
        "status": "error",
        "message": f"Preprocessed resume text file not found: {txt_path}. Run text_cleaner_detector.py first.",
        "code": 502
    }
    print(json.dumps(error_response))
    sys.exit()

with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove empty lines and strip
lines = [line.strip() for line in lines if line.strip()]

print(f"Loaded {len(lines)} non-empty lines from {txt_path}\n")
print("="*80)

# Store all extracted skills and knowledge
all_skills = []
all_knowledge = []
processed_entities = set()  # To avoid duplicates from overlapping windows

# Process lines using sliding window
print(f"Processing {len(lines)} lines with sliding window (size={WINDOW_SIZE})...\n")

for window_start in range(0, len(lines), 1):  # Slide by 1 line each time
    # Get window of lines
    window_end = min(window_start + WINDOW_SIZE, len(lines))
    window_lines_indices = list(range(window_start, window_end))
    
    if len(window_lines_indices) < 1:
        break
    
    # Get the actual lines
    window_lines_text = [lines[i] for i in window_lines_indices]
    
    # Combine lines with space separator and track line start positions
    combined_text = ' '.join(window_lines_text)
    
    # Preprocess combined_text: Replace hyphens, long hyphens, and parentheses with spaces
    # This must be done BEFORE passing to extractors so model can recognize hyphenated words
    combined_text = combined_text.replace('-', ' ').replace('—', ' ').replace('–', ' ').replace('(', ' ').replace(')', ' ')
    # Clean up multiple spaces
    combined_text = ' '.join(combined_text.split())
    
    # Calculate where each line starts in the combined text
    line_starts = [0]
    current_pos = 0
    for line_text in window_lines_text[:-1]:
        current_pos += len(line_text) + 1  # +1 for space
        line_starts.append(current_pos)
    
    # Convert to 1-indexed line numbers for display
    window_line_nums = [i + 1 for i in window_lines_indices]
    
    print(f"\n{'='*80}")
    print(f"Processing Window: Lines {window_line_nums[0]}-{window_line_nums[-1]}")
    print(f"{'='*80}")
    print(f"Combined text: {combined_text[:150]}{'...' if len(combined_text) > 150 else ''}\n")
    
    # Check token limits
    skill_within_limit, skill_token_count = check_token_limit(combined_text, skill_tokenizer, MAX_TOKENS)
    knowledge_within_limit, knowledge_token_count = check_token_limit(combined_text, knowledge_tokenizer, MAX_TOKENS)
    
    if not skill_within_limit or not knowledge_within_limit:
        print(f"⚠ WARNING: Window exceeds token limit!")
        print(f"  Skill tokens: {skill_token_count}/{MAX_TOKENS}")
        print(f"  Knowledge tokens: {knowledge_token_count}/{MAX_TOKENS}")
        print(f"  Skipping this window...")
        continue
    
    # Extract Skills
    skill_results = skill_extractor(combined_text)
    merged_skills = merge_bio_entities(skill_results, combined_text)
    merged_skills = merge_adjacent_entities(merged_skills, combined_text)
    merged_skills = split_entities_by_separators(merged_skills, combined_text)
    merged_skills = filter_by_confidence(merged_skills, CONFIDENCE_THRESHOLD)
    
    if merged_skills:
        print(f"  Skills found: {len(merged_skills)}")
        for entity in merged_skills:
            cleaned_text = clean_entity_text(entity['text'])
            if cleaned_text and is_valid_entity(cleaned_text):
                # Capitalize every word for display
                capitalized_text = capitalize_words(cleaned_text)
                
                # Determine primary line
                primary_line = determine_primary_line(
                    entity['start'], entity['end'], 
                    line_starts, window_line_nums
                )
                
                # Create unique key for deduplication (lowercase text + primary line)
                entity_key = (capitalized_text.lower(), primary_line)
                
                if entity_key not in processed_entities:
                    processed_entities.add(entity_key)
                    print(f"    - '{capitalized_text}' (Line {primary_line}, Score: {entity['score']:.4f})")
                    all_skills.append({
                        'text': capitalized_text,
                        'score': entity['score'],
                        'line': primary_line
                    })
                else:
                    print(f"    - '{capitalized_text}' (Line {primary_line}, Score: {entity['score']:.4f}) [DUPLICATE - SKIPPED]")
    else:
        print("  No skills found.")
    
    # Extract Knowledge
    knowledge_results = knowledge_extractor(combined_text)
    merged_knowledge = merge_bio_entities(knowledge_results, combined_text)
    merged_knowledge = merge_adjacent_entities(merged_knowledge, combined_text)
    merged_knowledge = split_entities_by_separators(merged_knowledge, combined_text)
    merged_knowledge = filter_by_confidence(merged_knowledge, CONFIDENCE_THRESHOLD)
    
    if merged_knowledge:
        print(f"  Knowledge found: {len(merged_knowledge)}")
        for entity in merged_knowledge:
            cleaned_text = clean_entity_text(entity['text'])
            if cleaned_text and is_valid_entity(cleaned_text):
                # Capitalize every word for display
                capitalized_text = capitalize_words(cleaned_text)
                
                # Determine primary line
                primary_line = determine_primary_line(
                    entity['start'], entity['end'], 
                    line_starts, window_line_nums
                )
                
                # Create unique key for deduplication
                entity_key = (capitalized_text.lower(), primary_line)
                
                if entity_key not in processed_entities:
                    processed_entities.add(entity_key)
                    print(f"    - '{capitalized_text}' (Line {primary_line}, Score: {entity['score']:.4f})")
                    all_knowledge.append({
                        'text': capitalized_text,
                        'score': entity['score'],
                        'line': primary_line
                    })
                else:
                    print(f"    - '{capitalized_text}' (Line {primary_line}, Score: {entity['score']:.4f}) [DUPLICATE - SKIPPED]")
    else:
        print("  No knowledge found.")

all_skills = deduplicate_entities(all_skills)
all_knowledge = deduplicate_entities(all_knowledge)

# Remove substring duplicates from same line
print("\n" + "="*80)
print("Removing substring duplicates...")
print("="*80)
all_skills = remove_substring_duplicates(all_skills)
all_knowledge = remove_substring_duplicates(all_knowledge)

# Sort by score (descending)
all_skills.sort(key=lambda x: x['score'], reverse=True)
all_knowledge.sort(key=lambda x: x['score'], reverse=True)

# Final Summary
print("\n" + "="*80)
print("\n--- FINAL SUMMARY ---")
print(f"Confidence threshold applied: {CONFIDENCE_THRESHOLD}")
print(f"Sliding window size: {WINDOW_SIZE} lines")
print(f"\nTotal Skills Extracted: {len(all_skills)}")
if all_skills:
    for skill in all_skills:
        print(f"  - '{skill['text']}' (Line {skill['line']}, Score: {skill['score']:.4f})")

print(f"\nTotal Knowledge Extracted: {len(all_knowledge)}")
if all_knowledge:
    for knowledge in all_knowledge:
        print(f"  - '{knowledge['text']}' (Line {knowledge['line']}, Score: {knowledge['score']:.4f})")

print("\n" + "="*80)
print("\nExtraction complete!")