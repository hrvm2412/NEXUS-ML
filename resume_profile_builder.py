import json
import os
import sys

# Placeholder variables — to be replaced with actual database values later
EDUCATION_DEGREE : str                   = "BS Electronics Engineering"
OCCUPATION_TITLES: list[tuple[str, int]] = [
    ("Electronics Engineer", 2),    # (title, years of experience)
    ("Broadcast Technician", 1),
]

def build_profile_text(
  education_degree : str,
  occupation_titles: list[tuple[str, int]],
  skills           : list[str],
  knowledge        : list[str],
) -> str           : 
    occupation_part       = format_occupation_sentence(occupation_titles)
    skills_knowledge_part = format_skills_and_knowledge(skills, knowledge)

    return f"{education_degree}. {occupation_part}. {skills_knowledge_part}."

def extract_texts(entries: list[dict]) -> list[str]:
    """Extract only the text values from a list of scored entries"""
    return [entry["text"] for entry in entries]

def format_occupation_sentence(titles: list[tuple[str, int]]) -> str:
    parts = [f"{title} with {years} year(s) of experience" for title, years in titles]
    return "Experienced: " + ", ".join(parts)

def format_skills_and_knowledge(skills: list[str], knowledge: list[str]) -> str:
    skills_str    = ", ".join(skills) if skills else "N/A"
    knowledge_str = ", ".join(knowledge) if knowledge else "N/A"
    return f"Soft Skills: {skills_str}. Technical Knowledge: {knowledge_str}"

def load_json(filepath: str) -> dict:
    with open(filepath, "r", encoding = "utf-8") as f:
        return json.load(f)

def parse_profile(filepath: str) -> str:
    data = load_json(filepath)

    skills    = extract_texts(data.get("skills", []))
    knowledge = extract_texts(data.get("knowledge", []))

    return build_profile_text(
        education_degree  = EDUCATION_DEGREE,
        occupation_titles = OCCUPATION_TITLES,
        skills            = skills,
        knowledge         = knowledge,
    )

if __name__     == "__main__":
    filepath      = "Extracted_skills_knowledge/Resume_skills_knowledge.json"

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"Resume skills JSON not found: {filepath}",
            "code"   : 500
        }
        print(json.dumps(error_response))
        sys.exit()

    profile_text  = parse_profile(filepath)

    output_dir  = "Profile_text"
    output_file = os.path.join(output_dir, "resume_profile.txt")
    os.makedirs(output_dir, exist_ok = True)

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(profile_text)

    print("JOB SEEKER PROFILE TEXT")
    print(profile_text)
    print(f"\nSaved to: {output_file}")