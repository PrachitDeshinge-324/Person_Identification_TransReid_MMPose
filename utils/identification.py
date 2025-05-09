import numpy as np
import pickle

def load_identity_database(db_path):
    with open(db_path, "rb") as f:
        return pickle.load(f)

def identify_person(query_feat, identity_db, threshold=0.5):
    """
    query_feat: np.ndarray, feature vector of detected person
    identity_db: dict, {person_id: {"appearance": feature_vector}}
    Returns: best_match_id, best_score
    """
    best_id = None
    best_score = -1
    for person_id, feats in identity_db.items():
        db_feat = feats["appearance"]
        if db_feat is None or query_feat is None:
            continue
        # Cosine similarity
        sim = np.dot(query_feat, db_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(db_feat) + 1e-6)
        if sim > best_score:
            best_score = sim
            best_id = person_id
    if best_score >= threshold:
        return best_id, best_score
    else:
        return None, best_score
