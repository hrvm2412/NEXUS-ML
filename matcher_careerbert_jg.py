import json
import numpy as np
import sys
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm.auto import tqdm

# Load the model
MODEL_ID         = "lwolfrum2/careerbert-jg"
LOCAL_MODEL_PATH = Path("Models/careerbert-jg")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Load profile texts from files
POSTING_PATH = Path("Profile_text/job_posting_profile.txt")
RESUME_PATH  = Path("Profile_text/resume_profile.txt")

def encode(jobbert_model: SentenceTransformer, texts: list[str], batch_size: int = 8) -> np.ndarray:
    return jobbert_model.encode(texts, batch_size = batch_size, show_progress_bar = True, convert_to_numpy = True)

def find_top_matches(query_title: str, candidate_titles: list, top_k: int = 3):
    """Find the top-k most similar job titles to a given query"""
    all_titles  = [query_title] + candidate_titles
    embeddings  = encode(model, all_titles)

    query_emb      = embeddings[0:1]
    candidate_embs = embeddings[1:]

    scores      = cos_sim(query_emb, candidate_embs)[0]
    top_indices = torch.argsort(scores, descending = True)[:top_k]

    return [(candidate_titles[i], round(scores[i].item(), 4)) for i in top_indices]

def load_model(model_id: str, local_path: Path, device: str) -> SentenceTransformer:
    if local_path.exists():
        return SentenceTransformer(str(local_path), device = device)
    
    local_path.mkdir(parents = True, exist_ok = True)
    loaded_model = SentenceTransformer(model_id, device=device)
    loaded_model.save(str(local_path))
    return loaded_model

def truncate_to_max_tokens(jobbert_model: SentenceTransformer, text: str) -> str:
    """Truncate text to the model's maximum token limit, then decode back to string"""
    tokenizer  = jobbert_model.tokenizer
    max_tokens = jobbert_model.max_seq_length
    tokens     = tokenizer(text, truncation = True, max_length = max_tokens, return_tensors = "pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens = True)

if __name__ == "__main__":

    model = load_model(MODEL_ID, LOCAL_MODEL_PATH, DEVICE)

    try:
        if not POSTING_PATH.exists():
            raise FileNotFoundError(f"Job posting profile not found: {POSTING_PATH}")
        if not RESUME_PATH.exists():
            raise FileNotFoundError(f"Resume profile not found: {RESUME_PATH}")
    except FileNotFoundError as e:
        error_response = {
            "status" : "error",
            "message": str(e),
            "code"   : 700
        }
        print(json.dumps(error_response))
        sys.exit()

    with open(POSTING_PATH, "r", encoding = "utf-8") as f:
        posting_text = truncate_to_max_tokens(model, f.read().strip())

    with open(RESUME_PATH, "r", encoding = "utf-8") as f:
        resume_text = truncate_to_max_tokens(model, f.read().strip())

    # Compute similarity between job posting and resume
    print("JOB POSTING VS RESUME SIMILARITY")
    print(f"\n  Job Posting : {posting_text}")
    print(f"\n  Resume      : {resume_text}")

    embeddings = encode(model, [posting_text, resume_text])
    score      = cos_sim(embeddings[0:1], embeddings[1:2])[0][0].item()

    print(f"\n  Similarity Score: {round(score, 4)}")