import json
import re


def normalize_text(text: str):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def exact_match(predicted: str, expected: str):
    return normalize_text(predicted) == normalize_text(expected)

def keyword_coverage(predicted: str, keywords: List[str]):
    predicted_norm = normalize_text(predicted)
    found = sum(1 for keyword in keywords if normalize_text(keyword) in predicted_norm)
    return found / len(keywords) if keywords else 0.0

def retrieval_hit_at_k(retrieved_chunks: List[str], expected_chunk_substring: str):
    expected_norm = normalize_text(expected_chunk_substring)
    return any(expected_norm in normalize_text(chunk) for chunk in retrieved_chunks)

def load_dataset(path: str):
    with open(path, "r") as f:
        return json.load(f)
