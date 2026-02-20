import torch
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device, cos_sim


# Load the model
model = SentenceTransformer("TechWolf/JobBERT-v3")

def encode_batch(jobbert_model, texts, text_key: str = "anchor"):
    features = jobbert_model.tokenize(texts)
    features = batch_to_device(features, jobbert_model.device)
    features["text_keys"] = [text_key]

    with torch.no_grad():
        out_features = jobbert_model.forward(features)

    return out_features["sentence_embedding"].cpu().numpy()

def encode(jobbert_model, texts, batch_size: int = 8, text_key: str = "anchor"):
    # Sort texts by length and keep track of original indices
    sorted_indices = np.argsort([len(text) for text in texts])
    sorted_texts = [texts[i] for i in sorted_indices]

    embeddings = []

    # Encode in batches
    for i in tqdm(range(0, len(sorted_texts), batch_size), desc="Encoding"):
        batch = sorted_texts[i:i + batch_size]
        embeddings.append(encode_batch(jobbert_model, batch, text_key))

    # Concatenate embeddings and reorder to original indices
    sorted_embeddings = np.concatenate(embeddings)
    original_order = np.argsort(sorted_indices)

    return sorted_embeddings[original_order]

def find_top_matches(query_title: str, candidate_titles: list, top_k: int = 3):
    """Find the top-k most similar job titles to a given query."""
    all_titles = [query_title] + candidate_titles
    embeddings = encode(model, all_titles)

    query_emb = embeddings[0:1]
    candidate_embs = embeddings[1:]

    scores = cos_sim(query_emb, candidate_embs)[0]
    top_indices = torch.argsort(scores, descending=True)[:top_k]

    return [(candidate_titles[i], round(scores[i].item(), 4)) for i in top_indices]

if __name__ == "__main__":
    # --- Similarity Matrix Demo ---
    job_titles = [
        "Software Engineer",
        "高级软件开发人员",   # senior software developer (Chinese)
        "Produktmanager",    # product manager (German)
        "Científica de datos"  # data scientist (Spanish)
    ]

    print("=== Similarity Matrix ===")
    embeddings = encode(model, job_titles)
    similarities = cos_sim(embeddings, embeddings)

    for i, title_a in enumerate(job_titles):
        for j, title_b in enumerate(job_titles):
            if j > i:
                score = round(similarities[i][j].item(), 4)
                print(f"  {title_a!r} <-> {title_b!r} : {score}")

    # --- Top Match Demo ---
    print("\n=== Top Matches for 'Data Analyst' ===")
    query = "Data Analyst"
    candidates = [
        "Business Intelligence Developer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Financial Analyst",
        "Software Architect"
    ]

    matches = find_top_matches(query, candidates, top_k=3)
    for rank, (title, score) in enumerate(matches, 1):
        print(f"  {rank}. {title!r} — score: {score}")