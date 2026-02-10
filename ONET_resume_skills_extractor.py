import csv
import json
import os
import spacy
import sys

def extract_text(tech_skills_path, resume_prep_path, nlp):
    """
    Extract skills from preprocessed resume using ONET data
    """
    # STEP 1: Read Resume_preprocessed.txt
    with open(resume_prep_path, "r", encoding = "utf-8") as f:
        text = f.read()
 
    # STEP 2: Load patterns from CSVs
    print("Loading ONET data ...")

    patterns         = []
    display_patterns = {}

    with open(tech_skills_path, "r", encoding = "utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("Example"):
                # Tokenize to match preprocessed text (spaced tokens)
                p_label = " ".join([t.text.lower() for t in nlp.make_doc(row["Example"].strip()) if not t.is_space])

                # Original casing preserved for JSON output
                p_text  = row["Example"].strip()                 
                desc    = row.get("Commodity Title", "").strip()

                if p_label:
                    patterns.append({"label": "SKILL", "pattern": p_label})

                    # Overwrite if exists
                    display_patterns[p_label] = {
                        "description": desc,
                        "display"    : p_text
                    }
    
    # STEP 3: Reset and add EntityRuler to pipeline
    print("Adding EntityRuler to pipeline ...")
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")

    ruler = nlp.add_pipe("entity_ruler", before = "ner")
    ruler.add_patterns(patterns)

    # STEP 4: Extract Skills
    print("Extracting skills ...")
    doc = nlp(text)

    extracted_skills_list = []
    seen_skills           = set()

    for ent in doc.ents:
        if ent.label_ == "SKILL":
            skill_data  = display_patterns.get(ent.text, {})
            skill_name  = skill_data.get("display", ent.text)

            if skill_name not in seen_skills: 
                seen_skills.add(skill_name)
                
                extracted_skills_list.append({
                    "Skill"      : skill_name,
                    "Description": skill_data.get("description", "")
                })

    # STEP 5: Save to JSON
    output_data = {"ONET Technology Skills": extracted_skills_list}
    output_path = os.path.join(os.path.dirname(resume_prep_path), "Output/ONET_tech_skills_extracted.json")

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(output_data, f, indent = 4)
    print(f"Skills saved to: {output_path}")

def init_extractor(nlp = None):
    if nlp is None:
        print("Loading spaCy model ... ")

        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            # Sample error to frontend
            error_response = {
                "status" : "error",
                "message": "SpaCy model 'en_core_web_lg' not found.",
                "code"   : 500
            }
            print(json.dumps(error_response))
            sys.exit()

        print("Model loaded.")
        return nlp

# START
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading ONET data and preprocessed resume ...")
    
    tech_skills_path = os.path.join(script_dir, "ONET_input/ONET_technology_skills_formatted.csv")
    resume_prep_path = os.path.join(script_dir, "Resume_preprocessed.txt")

    try:
        if not os.path.exists(tech_skills_path) or not os.path.exists(resume_prep_path):
            raise FileNotFoundError
    except FileNotFoundError:
        # Sample error to frontend
        error_response = {
            "status" : "error",
            "message": "ONET data or preprocessed resume not found.",
            "code"   : 501
        }
        print(json.dumps(error_response))
        sys.exit()

    print("Data loaded.")
    nlp = init_extractor()
    extract_text(tech_skills_path, resume_prep_path, nlp)