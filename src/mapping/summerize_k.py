# src/eval/summarize_k.py
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def compute_entropy(distribution: Dict[int, int]) -> float:
    """Shannon entropy (base-2) of a count distribution."""
    counts = np.array(list(distribution.values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def summarize_from_csv(file_path: str) -> Dict[str, float]:
    """
    Read a cluster-mapping CSV and compute:
      - avg_top_mapping_ratio: average max(cluster_count)/sum(cluster_count)
      - avg_entropy: average entropy across rows
    The CSV is expected to have columns ending with '_count'.
    """
    df = pd.read_csv(file_path)
    cluster_cols = [c for c in df.columns if c.endswith("_count")]

    top_ratios: List[float] = []
    entropies: List[float] = []

    for _, row in df.iterrows():
        distribution = {
            int(col.split("_")[2]): int(row[col])
            for col in cluster_cols
            if pd.notna(row[col])
        }
        if not distribution:
            continue

        total = sum(distribution.values())
        top_ratio = max(distribution.values()) / total if total > 0 else 0.0
        entropy = compute_entropy(distribution)

        top_ratios.append(top_ratio)
        entropies.append(entropy)

    return {
        "file": os.path.basename(file_path),
        "avg_top_mapping_ratio": round(float(np.mean(top_ratios)), 4) if top_ratios else 0.0,
        "avg_entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
    }


def summarize_k_range(
    base_dir: str,
    prefix: str = "labse_k_",
    k_min: int = 3,
    k_max: int = 13,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch summarize files like '{prefix}{k}.csv' for k in [k_min, k_max].
    Returns a DataFrame with columns: [file, avg_top_mapping_ratio, avg_entropy, k].
    Optionally write the summary to 'out_csv'.
    """
    results: List[Dict[str, float]] = []

    for k in range(k_min, k_max + 1):
        file_path = os.path.join(base_dir, f"{prefix}{k}.csv")
        if os.path.exists(file_path):
            summary = summarize_from_csv(file_path)
            summary["k"] = k
            results.append(summary)

    df_summary = pd.DataFrame(results)
    if out_csv:
        df_summary.to_csv(out_csv, index=False)
    return df_summary


