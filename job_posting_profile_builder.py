import json
import os
import sys

# Placeholder variables — to be replaced with actual database values later
EDUCATION_DEGREE_REQUIRED: str             = "Bachelor of Science in Electronics Engineering"
OCCUPATION_TITLE         : tuple[str, int] = ("Electronics Engineer", 2)                       # (title, years of experience required)

def build_posting_text(
  education_degree: str,
  occupation_title: tuple[str, int],
  skills          : list[str],
  knowledge       : list[str],
) -> str          : 
    education_part        = format_education_sentence(education_degree)
    occupation_part       = format_occupation_sentence(*occupation_title)
    skills_knowledge_part = format_skills_and_knowledge(skills, knowledge)

    return f"{education_part}. {occupation_part}. {skills_knowledge_part}."

def extract_texts(entries: list[dict]) -> list[str]:
    """Extract only the text values from a list of scored entries"""
    return [entry["text"] for entry in entries]

def format_education_sentence(degree: str) -> str:
    return f"Requires a {degree} or equivalent"

def format_occupation_sentence(title: str, years: int) -> str:
    return f"Looking for: {title} with {years} year(s) of experience"

def format_skills_and_knowledge(skills: list[str], knowledge: list[str]) -> str:
    skills_str    = ", ".join(skills) if skills else "N/A"
    knowledge_str = ", ".join(knowledge) if knowledge else "N/A"
    return f"Soft Skills: {skills_str}. Technical Knowledge: {knowledge_str}"

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding = "utf-8") as f:
        return json.load(f)

def parse_posting(filepath: str) -> str:
    data = load_json(filepath)

    skills    = extract_texts(data.get("skills", []))
    knowledge = extract_texts(data.get("knowledge", []))

    return build_posting_text(
        education_degree = EDUCATION_DEGREE_REQUIRED,
        occupation_title = OCCUPATION_TITLE,
        skills           = skills,
        knowledge        = knowledge,
    )

if __name__     == "__main__":
    filepath      = "Extracted_skills_knowledge/Job_posting_skills_knowledge.json"

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Job posting skills JSON not found: {filepath}",
            "code"   : 600
        }
        print(json.dumps(error_response))
        sys.exit()

    posting_text  = parse_posting(filepath)

    output_dir  = "Profile_text"
    output_file = os.path.join(output_dir, "job_posting_profile.txt")
    os.makedirs(output_dir, exist_ok = True)

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(posting_text)

    print("JOB POSTING PROFILE TEXT")
    print(posting_text)
    print(f"\nSaved to: {output_file}")