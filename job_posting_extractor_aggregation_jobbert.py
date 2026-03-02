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

# Placeholder variable — to be replaced with actual database value later
JOB_POSTING_TEXT = """
Electronics Engineer with experience in embedded programming and circuit analysis.
Proficiency in C/C++ and Python is required.
Knowledge of Proteus, Multisim, and Cisco Packet Tracer is an advantage.
Must have strong communication and problem-solving skills.
"""

def capitalize_words(text):
    words = text.split()
    capitalized_words = []
    for word in words:
        if any(c.isupper() for c in word):
            capitalized_words.append(word)
        else:
            capitalized_words.append(word.capitalize())
    return ' '.join(capitalized_words)

def check_token_limit(text, tokenizer, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens = True)
    return len(tokens) <= max_tokens, len(tokens)

def clean_entity_text(text):
    cleaned = re.sub(r'[(){}\[\]]', '', text).strip()
    while cleaned and cleaned[-1] in ',.;:!?':
        cleaned = cleaned[:-1].strip()
    while cleaned and cleaned[0] in ',.;:!?':
        cleaned = cleaned[1:].strip()
    return cleaned

def deduplicate_entities(entities):
    seen = {}
    for entity in entities:
        text_lower = entity['text'].lower()
        if text_lower not in seen or entity['score'] > seen[text_lower]['score']:
            seen[text_lower] = entity
    return list(seen.values())

def determine_primary_line(entity_start, entity_end, line_starts, window_lines):
    line_contributions = []
    for idx, (line_num, line_start) in enumerate(zip(window_lines, line_starts)):
        if idx < len(line_starts) - 1:
            line_end = line_starts[idx + 1]
        else:
            line_end = float('inf')
        overlap_start = max(entity_start, line_start)
        overlap_end   = min(entity_end, line_end)
        overlap       = max(0, overlap_end - overlap_start)
        line_contributions.append((line_num, overlap))
    primary_line = max(line_contributions, key = lambda x: x[1])[0]
    return primary_line

def download_model_if_needed(model_path, model_id):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Downloading {model_id} from HuggingFace ...")
        try:
            os.makedirs(model_path, exist_ok = True)
            model     = AutoModelForTokenClassification.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
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
                "code"   : 400
            }
            print(json.dumps(error_response))
            sys.exit()
    else:
        print(f"Model found at {model_path}")

def expand_to_word_boundaries(text, start, end):
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in ',.;:!?()[]{}/-':
        start -= 1
    while end < len(text) and not text[end].isspace() and text[end] not in ',.;:!?()[]{}/-':
        end += 1
    return start, end

def filter_by_confidence(entities, threshold = CONFIDENCE_THRESHOLD):
    return [e for e in entities if e['score'] >= threshold]

def is_valid_entity(text):
    if not text or len(text) < 2:
        return False
    if not any(c.isalnum() for c in text):
        return False
    if text.strip() in ['(', ')', '()', '[]', '{}', '( )', '[ ]', '{ }']:
        return False
    return True

def merge_adjacent_entities(entities, original_text, max_gap = 2):
    if not entities:
        return []
    merged = []
    i      = 0
    while i < len(entities):
        current = entities[i]
        while i + 1 < len(entities):
            next_entity  = entities[i + 1]
            gap          = next_entity['start'] - current['end']
            if gap <= max_gap:
                between_text  = original_text[current['end']:next_entity['start']]
                if between_text.strip() == '':
                    has_separator = any(sep in between_text for sep in ',;:|')
                    if not has_separator:
                        current['text']  = original_text[current['start']:next_entity['end']]
                        current['end']   = next_entity['end']
                        current['score'] = (current['score'] + next_entity['score']) / 2
                        i               += 1
                    else:
                        break
                else:
                    break
            else:
                break
        merged.append(current)
        i += 1
    return merged

def merge_bio_entities(results, original_text):
    if not results:
        return []
    merged         = []
    current_entity = None
    for entity in results:
        label = entity['entity_group']
        if label == 'B':
            if current_entity:
                merged.append(current_entity)
            start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
            current_entity = {
                'text' : original_text[start:end],
                'start': start,
                'end'  : end,
                'score': entity['score']
            }
        elif label == 'I' and current_entity:
            between_text = original_text[current_entity['end']:entity['start']]
            if any(sep in between_text for sep in [',', ';', '|', '&']):
                merged.append(current_entity)
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                current_entity = {
                    'text' : original_text[start:end],
                    'start': start,
                    'end'  : end,
                    'score': entity['score']
                }
            else:
                start, end = expand_to_word_boundaries(original_text, entity['start'], entity['end'])
                current_entity['text']  = original_text[current_entity['start']:end]
                current_entity['end']   = end
                current_entity['score'] = (current_entity['score'] + entity['score']) / 2
    if current_entity:
        merged.append(current_entity)
    return merged

def remove_substring_duplicates(entities):
    if not entities:
        return []
    by_line = {}
    for entity in entities:
        line = entity['line']
        if line not in by_line:
            by_line[line] = []
        by_line[line].append(entity)
    filtered = []
    for line_num, line_entities in by_line.items():
        line_entities.sort(key = lambda x: len(x['text']), reverse = True)
        keep = []
        for i, entity in enumerate(line_entities):
            is_substring = False
            for kept_entity in keep:
                if entity['text'].lower() in kept_entity['text'].lower():
                    is_substring = True
                    break
            if not is_substring:
                keep.append(entity)
        filtered.extend(keep)
    return filtered

def split_entities_by_separators(entities, original_text):
    split_entities = []
    for entity in entities:
        text = entity['text']
        if any(sep in text for sep in [',', ';', '|']):
            parts       = re.split(r'[,;|]', text)
            current_pos = entity['start']
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                part_start = original_text.find(part, current_pos)
                if part_start != -1:
                    part_end = part_start + len(part)
                    split_entities.append({
                        'text' : part,
                        'start': part_start,
                        'end'  : part_end,
                        'score': entity['score']
                    })
                    current_pos = part_end
        else:
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
        
        skill_tokenizer     = AutoTokenizer.from_pretrained(SKILL_MODEL_PATH)
        knowledge_tokenizer = AutoTokenizer.from_pretrained(KNOWLEDGE_MODEL_PATH)
        
        print("Models loaded successfully from local storage.")
    except Exception as e:
        error_response = {
            "status" : "error",
            "message": f"Failed to load models: {str(e)}",
            "code"   : 401
        }
        print(json.dumps(error_response))
        sys.exit()

    MAX_TOKENS = min(skill_tokenizer.model_max_length, knowledge_tokenizer.model_max_length)

    print(f"Model max tokens: {MAX_TOKENS}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Sliding window size: {WINDOW_SIZE} lines")
    print("Pipelines initialized successfully.")

    # 3. Load job posting text from variable
    lines = [line.strip() for line in JOB_POSTING_TEXT.splitlines() if line.strip()]

    # Store all extracted skills and knowledge
    all_skills         = []
    all_knowledge      = []
    processed_entities = set()

    # 4. Process lines using sliding window
    print(f"Processing {len(lines)} lines with sliding window (size = {WINDOW_SIZE}) ...")

    for window_start in range(0, len(lines), 1):
        window_end           = min(window_start + WINDOW_SIZE, len(lines))
        window_lines_indices = list(range(window_start, window_end))
        
        if len(window_lines_indices) < 1:
            break
        
        window_lines_text = [lines[i] for i in window_lines_indices]
        combined_text     = '\n'.join(window_lines_text)
        
        line_starts = [0]
        current_pos = 0
        for line_text in window_lines_text[:-1]:
            current_pos += len(line_text) + 1
            line_starts.append(current_pos)
        
        window_line_nums = [i + 1 for i in range(window_start, window_end)]
        del window_lines_text, current_pos

        skill_within_limit    , skill_token_count     = check_token_limit(combined_text, skill_tokenizer, MAX_TOKENS)
        knowledge_within_limit, knowledge_token_count = check_token_limit(combined_text, knowledge_tokenizer, MAX_TOKENS)
        
        if not skill_within_limit or not knowledge_within_limit:
            error_response = {
                "status" : "error",
                "message": f"Text window exceeds model token limit ({MAX_TOKENS}).",
                "code"   : 403
            }
            print(json.dumps(error_response))
            sys.exit(1)
        
        # 5. Extract Skills & Knowledge
        skill_results = skill_extractor(combined_text)
        merged_skills = merge_bio_entities(skill_results, combined_text)
        merged_skills = merge_adjacent_entities(merged_skills, combined_text)
        merged_skills = split_entities_by_separators(merged_skills, combined_text)
        merged_skills = filter_by_confidence(merged_skills, CONFIDENCE_THRESHOLD)
        
        for entity in skill_results:
            cleaned_text = clean_entity_text(entity['text'])
            if cleaned_text and is_valid_entity(cleaned_text):
                capitalized_text = capitalize_words(cleaned_text)
                primary_line     = determine_primary_line(entity['start'], entity['end'], line_starts, window_line_nums)
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

        knowledge_results = knowledge_extractor(combined_text)
        merged_knowledge  = merge_bio_entities(knowledge_results, combined_text)
        merged_knowledge  = merge_adjacent_entities(merged_knowledge, combined_text)
        merged_knowledge  = split_entities_by_separators(merged_knowledge, combined_text)
        merged_knowledge  = filter_by_confidence(merged_knowledge, CONFIDENCE_THRESHOLD)
        
        for entity in merged_knowledge:
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

    # 6. Global Deduplication
    all_skills    = deduplicate_entities(all_skills)
    all_knowledge = deduplicate_entities(all_knowledge)

    # 7. Substring Cleanup
    print("Removing substring duplicates ...")
    all_skills    = remove_substring_duplicates(all_skills)
    all_knowledge = remove_substring_duplicates(all_knowledge)

    all_skills.sort(key = lambda x: x['score'], reverse = True)
    all_knowledge.sort(key = lambda x: x['score'], reverse = True)

    # 8. Save to Job_posting_skills_knowledge.json
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

    output_path = "Extracted_skills_knowledge/Job_posting_skills_knowledge.json"
    try:
        with open(output_path, 'w', encoding = 'utf-8') as f:
            json.dump(output, f, indent = 4, ensure_ascii = False)
    except Exception as e:
        error_response = {
            "status" : "error",
            "message": f"Failed to export JSON file: {str(e)}",
            "code"   : 404
        }
        print(json.dumps(error_response))
        sys.exit(1)

    print("Extraction complete!")