import hashlib
import tempfile
import os
import pandas as pd
from typing import Dict, List

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def save_temp_file(bytes_data: bytes, suffix: str = ".pdf") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(bytes_data)
    tmp.flush()
    tmp.close()
    return tmp.name

def remove_file_safe(path: str):
    try:
        os.remove(path)
    except Exception:
        pass

def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    # results is list of dicts; missing keys -> fill with None
    df = pd.DataFrame(results)
    # order columns if present
    cols = [
        "file_name", "title", "authors", "publication_year", "journal_or_conference",
        "research_objective", "methodology", "key_findings", "limitations", "error"
    ]
    existing = [c for c in cols if c in df.columns]
    # append any other columns
    other_cols = [c for c in df.columns if c not in existing]
    return df[existing + other_cols] if existing else df
